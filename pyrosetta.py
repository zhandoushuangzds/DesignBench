#!/usr/bin/env -S /bin/sh -c '"$(dirname $(readlink -f "$0"))/alphafold_ppi_shebang.sh" "$0" "$@"'

# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""


import sys
import os
# Optional: Only add site_packages if it exists and alphafold3 is not already importable
# This allows the script to work with installed alphafold3 packages
try:
    import alphafold3.test_af3_import
except ImportError:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.join(script_dir, 'site_packages')
    if os.path.exists(package_dir):
        sys.path.insert(0, package_dir)
        try:
            import alphafold3.test_af3_import
        except ImportError:
            raise ImportError(
                "alphafold3 not found. Please install alphafold3 or ensure "
                f"site_packages directory exists at {package_dir}"
            )


from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
import json
from typing import overload

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np


from Bio.PDB import PDBParser, MMCIFParser, MMCIFIO, PDBIO, PPBuilder
import io
import tempfile
import pyrosetta as pyro
import pyrosetta.rosetta as ros
pyro.init('-mute all')


_HOME_DIR = pathlib.Path(os.environ.get('HOME', os.path.expanduser('~')))

# Get defaults from environment variables, or use None to require explicit specification
_DEFAULT_MODEL_DIR_ENV = os.environ.get('AF3_MODEL_DIR')
_DEFAULT_DB_DIR_ENV = os.environ.get('AF3_DB_DIR')

_DEFAULT_MODEL_DIR = (
    pathlib.Path(_DEFAULT_MODEL_DIR_ENV) 
    if _DEFAULT_MODEL_DIR_ENV and os.path.exists(_DEFAULT_MODEL_DIR_ENV)
    else None
)
_DEFAULT_DB_DIR = (
    pathlib.Path(_DEFAULT_DB_DIR_ENV)
    if _DEFAULT_DB_DIR_ENV and os.path.exists(_DEFAULT_DB_DIR_ENV)
    else None
)

# bcov
_PDBS = flags.DEFINE_spaceseplist(
    'pdbs',
    None,
    'Path to pdbs to run AF3 on'
)

_SILENT_PATH = flags.DEFINE_string(
    'silent',
    None,
    'Path of an input silent file',
)

_CHAINBREAK_MODE = flags.DEFINE_string(
    'chainbreak_mode',
    'big_gap',
    'Choices are: big_gap, input_numbering, rough_guess',
)

_WRITE_ALL_STRUCTS = flags.DEFINE_bool(
    'write_all_structs',
    False,
    'Write the outputs from every diffusion trajectory',
)

_NUM_TEMPLATES = flags.DEFINE_integer(
    'num_templates',
    4,
    'How many templates to use?',
)

_TEMPLATE_FUZZ_ANG = flags.DEFINE_float(
    'template_fuzz_ang',
    0,
    'Add normally distributed noise with this as the std to the xyz of the extra templates',
)

_TEMPLATE_FUZZ_IDENT = flags.DEFINE_float(
    'template_fuzz_ident',
    1,
    'Corrupt extra template sequences such that they have this fractional identity to the start.',
)


_BUCKET_STEP = flags.DEFINE_integer(
  'bucket_step',
  16,
  'By default this is 256 (sort of a hidden variable) and so your 257aa input gets padded to 512aa!!!'
)

# /bcov




# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_FASTA = flags.DEFINE_string(
    'fasta',
    None,
    'Path to the input FASTA file.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './',
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix() if _DEFAULT_MODEL_DIR is not None else None,
    'Path to the model to use for inference. Can also be set via AF3_MODEL_DIR environment variable.',
)
_LIGANDS = flags.DEFINE_multi_string(
    'ligand',
    None,
    'Ligand names or SMILES. If ligand is on CCD then CCD_NAME3, otherwise SMILES. '
    'To add userCCD definition to a ligand, append a path to a CIF file with a comma to the ligand name.'
    'Example: --ligand CCD_HEM --ligand LIG,/path/to/LIG.cif'
)



# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths - check environment variables first, then system PATH
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    os.environ.get('JACKHMMER_BINARY_PATH', shutil.which('jackhmmer') or ''),
    'Path to the Jackhmmer binary. Can also be set via JACKHMMER_BINARY_PATH environment variable.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    os.environ.get('NHMMER_BINARY_PATH', shutil.which('nhmmer') or ''),
    'Path to the Nhmmer binary. Can also be set via NHMMER_BINARY_PATH environment variable.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    os.environ.get('HMMALIGN_BINARY_PATH', shutil.which('hmmalign') or ''),
    'Path to the Hmmalign binary. Can also be set via HMMALIGN_BINARY_PATH environment variable.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    os.environ.get('HMMSEARCH_BINARY_PATH', shutil.which('hmmsearch') or ''),
    'Path to the Hmmsearch binary. Can also be set via HMMSEARCH_BINARY_PATH environment variable.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    os.environ.get('HMMBUILD_BINARY_PATH', shutil.which('hmmbuild') or ''),
    'Path to the Hmmbuild binary. Can also be set via HMMBUILD_BINARY_PATH environment variable.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),) if _DEFAULT_DB_DIR is not None else None,
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order. Can also be set via AF3_DB_DIR environment variable.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Template search configuration.
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All '
    'templates released after this date will be ignored.',
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference. Defaults to the'
    ' 1st GPU on the system. Useful on multi-GPU systems to pin each run to a'
    ' specific GPU.',
)

_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    # ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
    #  '3584', '4096', '4608', '5120'],
    None,
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)
_SEEDS = flags.DEFINE_multi_integer(
    'seeds',
    [1],
    'Seed value(s) used for FASTA input. Ignored for all other input types.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.',
)

# Print debug flags
_PRINT_ALL_PAIRED_SPECIES = flags.DEFINE_bool(
    'print_all_paired_species',
    False,
    'If True, prints all species that are paired. Note that this may slightly'
    ' exceed the actual set of paired species in the featurized MSA, since this'
    ' will be cropped to max_paired_sequences.',
)


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
      return model.Model(self._model_config)(batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    identifier = self.model_params['__meta__']['__identifier__'].tobytes()
    result['__identifier__'] = identifier
    return result

  def extract_inference_results_and_maybe_embeddings(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> tuple[list[model.InferenceResult], dict[str, np.ndarray] | None]:
    """Extracts inference results and embeddings (if set) from model outputs."""
    inference_results = list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][:num_tokens]
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ]
    return inference_results, embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    print_all_paired_species: bool = False,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      conformer_max_iterations=conformer_max_iterations,
      print_all_paired_species=print_all_paired_species,
      max_templates=max(4, _NUM_TEMPLATES.value) # this used to be 4
  )
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key)
    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results, embeddings = (
        model_runner.extract_inference_results_and_maybe_embeddings(
            batch=example, result=result, target_name=fold_input.name
        )
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
    other_inputs: dict = None,
    start_time: int = 0
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  # output_terms = (
  #     pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  # ).read_text()

  # os.makedirs(output_dir, exist_ok=True)
  max_seed_idx = 'None'
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      seed_idx = f'{seed}-{sample_idx}'
      # if _WRITE_ALL_STRUCTS.value:
      addtl_ranking_score = write_ppi_result(result, other_inputs, tag=other_inputs['tag'] + f'_seed-{seed}_sample-{sample_idx}', 
                        seed_idx=seed_idx, start_time=start_time,
                        write=_WRITE_ALL_STRUCTS.value)
      # sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      # os.makedirs(sample_dir, exist_ok=True)
      # post_processing.write_output(
      #     inference_result=result, output_dir=sample_dir
      # )
      ranking_score = float(result.metadata['ranking_score']) + addtl_ranking_score
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result
        max_seed_idx = seed_idx

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(os.path.join(other_inputs['output_dir'], 'embeddings'))
      embeddings_name = f'{other_inputs["tag"]}-seed-{seed}_emb'
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings, output_dir=embeddings_dir, embeddings_name=embeddings_name
      )

  # if max_ranking_result is not None:  # True iff ranking_scores non-empty.
  #   post_processing.write_output(
  #       inference_result=max_ranking_result,
  #       output_dir=output_dir,
  #       # The output terms of use are the same for all seeds/samples.
  #       terms_of_use=output_terms,
  #       name=job_name,
  #   )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    # with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
    #   writer = csv.writer(f)
    #   writer.writerow(['seed', 'sample', 'ranking_score'])
    #   writer.writerows(ranking_scores)

  write_ppi_result(max_ranking_result, other_inputs, tag=other_inputs['tag'] + '_af3pred', seed_idx=max_seed_idx, start_time=start_time)

  with open(other_inputs['checkpoint'], 'a') as f:
    f.write(other_inputs['tag'] + '\n')


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  ...


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


### IK custom functions ###
def load_fold_inputs_from_list(input_dict_list: list):
  """
  Loads multiple fold inputs from a list of dicts.
  IK edit
  """

  fold_inputs = []
  for fold_job_idx, fold_job in enumerate(input_dict_list):
    # AlphaFold 3 JSON.
    try:
      fold_inputs.append(folding_input.Input.from_json(json.dumps(fold_job)))
    except ValueError as e:
      raise ValueError(
          f'Failed to load fold input from {fold_job}. The JSON'
          f' was detected to be an AlphaFold 3 JSON since the'
          ' top-level is not a list.'
      ) from e

  # folding_input.check_unique_sanitised_names(fold_inputs)
  return fold_inputs


def parse_fasta_input(_FASTA, _LIGANDS, _SEEDS):
    ff = open(_FASTA.value, "r").readlines()
    fasta = {l.strip().replace(">", ""): ff[i+1].strip() for i,l in enumerate(ff) if ">" in l}
    fasta = {k: v for k, v in sorted(fasta.items(), key=lambda item: len(item[1]), reverse=False)}  # re-ordering FASTA to start from the longest
    inputs_list = []
    for nam, seq in fasta.items():
        inpdict = {"name": nam,
                   "sequences": [], "modelSeeds": _SEEDS.value, "dialect": "alphafold3", "version": 1}
        for i,chseq in enumerate(seq.split(" ")):  # currently using ' ' to denote chain split
            ch_let = string.ascii_uppercase[i]
            _dct = {"protein": {"id": [ch_let],
                                "sequence": chseq,
                                "templates": [],
                                "unpairedMsa": "",
                                "pairedMsa": ""}}
            inpdict["sequences"].append(_dct)
        if _LIGANDS.value is not None:
            for j,lig in enumerate(_LIGANDS.value):
                ch_let = string.ascii_uppercase[i+j+1]
                _dct = {"ligand": {"id": [ch_let]}}
                if lig[:4] == "CCD_":
                    _dct["ligand"]["ccdCodes"] = [lig.replace("CCD_", "")]
                else:
                    if "," not in lig:
                      _dct["ligand"]["smiles"] = lig
                    else:
                      if ".sdf" in lig.split(",")[1]:
                        _dct["ligand"]["smiles"] = lig.split(",")[0]
                        _dct["ligand"]["sdf"] = lig.split(",")[1]
                      elif ".cif" in lig.split(",")[1]:
                        _dct["ligand"]["ccdCodes"] = [lig.split(",")[0]]
                        if "userCCD" not in inpdict:
                            inpdict["userCCD"] = ""
                        inpdict["userCCD"] += open(lig.split(",")[1], "r").read() + "\n"
                inpdict["sequences"].append(_dct)
        inputs_list.append(inpdict)
        print(inpdict)
    return inputs_list
### IK custom functions ###

### bcov custom functions ###


release_lines = [
        "# \n",
        "loop_\n",
        "_pdbx_audit_revision_history.ordinal \n",
        "_pdbx_audit_revision_history.data_content_type \n",
        "_pdbx_audit_revision_history.major_revision \n",
        "_pdbx_audit_revision_history.minor_revision \n",
        "_pdbx_audit_revision_history.revision_date \n",
        "1 'Structure model' 1 0 2004-07-13 \n",
        "2 'Structure model' 1 1 2011-06-14 \n",
        "3 'Structure model' 1 2 2011-07-13 \n",
        "4 'Structure model' 1 3 2011-07-27 \n",
        "5 'Structure model' 1 4 2012-12-12 \n",
        "6 'Structure model' 2 0 2023-08-23 \n",
        "# \n",
        "_diffrn_detector.diffrn_id              1 \n",
        "_diffrn_detector.detector               CCD \n",
        "_diffrn_detector.type                   'ADSC QUANTUM 4' \n",
        "_diffrn_detector.pdbx_collection_date   2003-05-05 \n",
        "_diffrn_detector.details                ? \n",
        "# \n",
        "_pdbx_database_status.status_code                     REL \n",
        "_pdbx_database_status.entry_id                        1T7D \n",
        "_pdbx_database_status.recvd_initial_deposition_date   2004-05-09 \n",
        "_pdbx_database_status.deposit_site                    RCSB \n",
        "_pdbx_database_status.process_site                    RCSB \n",
        "_pdbx_database_status.status_code_sf                  REL \n",
        "_pdbx_database_status.SG_entry                        . \n",
        "_pdbx_database_status.status_code_mr                  ? \n",
        "_pdbx_database_status.status_code_cs                  ? \n",
        "_pdbx_database_status.pdb_format_compatible           Y \n",
        "_pdbx_database_status.status_code_nmr_data            ? \n",
        "_pdbx_database_status.methods_development_category    ? \n",
        "# "
    ]




def create_ppi_json(model_pose, binder, target, binder_res_ids, target_res_ids, target_mmcif, other_inputs):

  # with open('target.cif', 'w') as f:
  #   f.write(target_mmcif)

  item = {
      "name": other_inputs['tag'],
      "sequences": [
        {
          "protein": {
            "id": "A",
            "sequence": binder.sequence(),
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": ""
          }
        },
        {
          "protein": {
            "id": "B",
            "sequence": target.sequence(),
            "unpairedMsa": "",
            "pairedMsa": "",
            "template_fuzz_ang": _TEMPLATE_FUZZ_ANG.value,
            "template_fuzz_ident": _TEMPLATE_FUZZ_IDENT.value,
            "templates": [
              {
                'mmcif': target_mmcif,
                "queryIndices": [i for i in range(target.size())],
                "templateIndices": [i for i in range(target.size())],
              }
            ]
          }
        }
      ],
      "modelSeeds": [1],
      "dialect": "alphafold3",
      "version": 1
      }

  if binder_res_ids is not None:
    item['sequences'][0]['protein']['res_id'] = [int(x) for x in binder_res_ids]
  if target_res_ids is not None:
    item['sequences'][1]['protein']['res_id'] = [int(x) for x in target_res_ids]

  for i in range(_NUM_TEMPLATES.value-1):
    item['sequences'][1]['protein']['templates'].append(item['sequences'][1]['protein']['templates'][0])


  return item


def mmcif_to_pose(mmcif_string):
  ''' Rosetta can read mmcif files from disk, but not from a string...'''

  from Bio.PDB import MMCIFParser, PDBIO
  parser = MMCIFParser(QUIET=True)
  inn = io.StringIO(mmcif_string)
  structure = parser.get_structure("protein", inn)
  pdbio = PDBIO()
  pdbio.set_structure(structure)

  out = io.StringIO()
  pdbio.save(out)

  pose = pyro.Pose()
  ros.core.import_pose.pose_from_pdbstring(pose, out.getvalue())

  return pose


def pose_to_mmcif(pose):
  ''' I just gave up on getting Rosetta to do this. Seems cif isn't very well supported'''

  ss = ros.std.stringstream()
  pose.dump_pdb(ss)
  pdbstring = ss.str()

  parser = PDBParser(QUIET=True)
  inn = io.StringIO(pdbstring)
  structure = parser.get_structure("protein", inn)
  mmio = MMCIFIO()
  mmio.set_structure(structure)

  out = io.StringIO()
  mmio.save(out)
  return out.getvalue() + ''.join(release_lines)

  return target_mmcif



def get_target_numbering(target, mode='big_gap', max_amid_distance=3, angstroms_per_aa=4):
  

  if mode == 'input_numbering':
    res_id = []
    for seqpos in range(1, target.size()+1):
      res_id.append(pose.pdb_info().number(seqpos))
    return res_id

  if mode == 'big_gap' or mode == 'rough_guess':
    res_id = []
    last_res_id = 0
    last_C_xyz = None
    for seqpos in range(1, target.size()+1):
      xyz = target.residue(seqpos).xyz('N')
      jump = False
      distance = 0
      if last_C_xyz is not None:
        distance = xyz.distance(last_C_xyz)
        jump = distance > 3

      if jump:
        if mode == 'big_gap':
          this_res_id = last_res_id + 201
        else:
          this_res_id = last_res_id + int(np.ceil( distance / angstroms_per_aa ))
      else:
        this_res_id = last_res_id + 1

      res_id.append(this_res_id)

      last_C_xyz = target.residue(seqpos).xyz('C')
      last_res_id = this_res_id

    return res_id

  assert False, 'Unknown --chainbreak_mode'



def pose_input_generator(pose, other_inputs):

  other_inputs['input_pose'] = pose

  assert pose.num_chains() == 2, 'Sorry! You need a 2-chain input. We can probably fix this if you have a good case'

  model_pose = pose.clone()
  binder, target = pose.split_by_chain()

  model_pose.pdb_info(ros.core.pose.PDBInfo(model_pose))
  other_inputs['model_pose'] = model_pose


  
  target_mmcif = pose_to_mmcif(target)

  binder_res_ids = None
  target_res_ids = get_target_numbering(target, mode=_CHAINBREAK_MODE.value)

  js = create_ppi_json(model_pose, binder, target, binder_res_ids, target_res_ids, target_mmcif, other_inputs)

  return other_inputs, folding_input.Input.from_json(json.dumps(js))

def pdb_input_generator(pdbs, other_inputs_in):

  for pdb in pdbs:
    other_inputs = {k:v for k,v in other_inputs_in.items()}
    other_inputs['tag'] = pdb_to_tag(pdb)
    pose = pyro.pose_from_file(pdb)

    yield pose_input_generator(pose, other_inputs)

def silent_input_generator(sfd_in, tags, other_inputs_in):

  for tag in tags:
    other_inputs = {k:v for k,v in other_inputs_in.items()}
    other_inputs['tag'] = tag

    pose = pyro.Pose()
    sfd_in.get_structure(tag).fill_pose(pose)

    yield pose_input_generator(pose, other_inputs)


def load_ppi_silent(silent_path, other_inputs, done):

  other_inputs['output_type'] = 'silent'
  sfd_in = ros.core.io.silent.SilentFileData(ros.core.io.silent.SilentFileOptions())
  sfd_in.read_file(silent_path)
  remaining_tags = []
  for tag in sfd_in.tags():
    if tag in done:
      print("Checkpoint: %s already done"%tag)
    else:
      remaining_tags.append(tag)

  if len(remaining_tags) == 0:
    return None
  else:
    return silent_input_generator(sfd_in, remaining_tags, other_inputs)

def pdb_to_tag(pdb):
  tag = os.path.basename(pdb)
  for suffix in ['.gz', '.cif', '.pdb']:
    if tag.endswith(suffix):
      tag = tag[:-len(suffix)]
  return tag


def load_ppi_pdbs(pdbs_in, other_inputs, done):

  pdbs = []
  for pdb in pdbs_in:
    tag = pdb_to_tag(pdb)
    if tag in done:
      print("Checkpoint: %s already done"%tag)
    else:
      pdbs.append(pdb)

  pdbs = [pdb for pdb in pdbs if pdb_to_tag(pdb) not in done]
  other_inputs['output_type'] = 'pdb'

  if len(pdbs) == 0:
    return None
  else:
    return pdb_input_generator(pdbs, other_inputs)


def load_ppi_inputs(output_dir, silent_path=None, pdbs=None):

  assert (silent_path is None) != (pdbs is None), 'Only specify one of --silent and --pdbs'

  other_inputs = {}
  other_inputs['checkpoint'] = os.path.join(output_dir, 'check.point')
  other_inputs['score_file'] = os.path.join(output_dir, 'out.sc')
  other_inputs['silent_file'] = os.path.join(output_dir, 'out.silent')
  other_inputs['output_dir'] = os.path.join(output_dir)

  done = set()
  if os.path.exists(other_inputs['checkpoint']):
    with open(other_inputs['checkpoint']) as f:
      for line in f:
        line = line.strip()
        if len(line) == 0:
          continue
        done.add(line)

  if 'DONE' in done:
    print("DONE in", other_inputs['checkpoint'])
    return None


  if silent_path is not None:
    return load_ppi_silent(silent_path, other_inputs, done)
  else:
    return load_ppi_pdbs(pdbs, other_inputs, done)


def get_final_dict(score_dict, string_dict):
  # print(score_dict)
  final_dict = {}
  keys_score = [] if score_dict is None else list(score_dict)
  keys_string = [] if string_dict is None else list(string_dict)

  all_keys = keys_score + keys_string

  argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

  for idx in argsort:
    key = all_keys[idx]

    if ( idx < len(keys_score) ):
            value = score_dict[key]
    else:
            value = string_dict[key]

    if ( not isinstance(value, str) ):
        value = "%8.3f"%value
    final_dict[key] = value

  return final_dict

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None):
  with open(scorefilename, "a") as f:
    add_to_score_file_open(tag, f, write_header, score_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
  final_dict = get_final_dict( score_dict, string_dict )
  if ( write_header ):
    f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
  scores_string = " ".join(final_dict.values())
  f.write("SCORE:     %s        %s\n"%(scores_string, tag))




def calculate_ppi_scores(processed_result, in_model_pose, out_model_pose, start_time=0, seed_idx='None'):
  score_dict = {}
  out_pose = out_model_pose.clone()

  monomer_size = in_model_pose.conformation().chain_end(1)

  # AF3 confidence values
  summary_js = json.loads(processed_result.structure_confidence_summary_json.decode('ascii'))
  confidence_js = json.loads(processed_result.structure_full_data_json.decode('ascii'))

  min_pair_pae = np.array(summary_js['chain_pair_pae_min'])
  chain_pair_iptm = np.array(summary_js['chain_pair_iptm'])
  mean_min_interface_pae = np.mean(min_pair_pae[0][1] + min_pair_pae[1][0])/2
  mean_pair_iptm = np.mean(chain_pair_iptm[0][1] + chain_pair_iptm[1][0])/2
  chain_a_ptm = summary_js['chain_ptm'][0]
  chain_b_ptm = summary_js['chain_ptm'][1]

  token_chain_ids = np.array(confidence_js['token_chain_ids'])
  assert (token_chain_ids[:monomer_size] == 'A').all()
  assert (token_chain_ids[monomer_size:] == 'B').all()
  interface_mask = (token_chain_ids[:, None] != token_chain_ids[None, :])
  token_pae = np.array(confidence_js['pae'])
  interface_pae = token_pae[interface_mask]
  mean_interface_pae = np.mean(interface_pae)

  score_dict['min_pae_interaction'] = mean_min_interface_pae
  score_dict['pae_interaction'] = mean_interface_pae
  score_dict['iptm'] = mean_pair_iptm
  score_dict['ptm_binder'] = chain_a_ptm
  score_dict['ptm_target'] = chain_b_ptm


  score_dict['binder_rmsd'] = ros.core.scoring.CA_rmsd(in_model_pose, out_model_pose, 1, monomer_size)
  score_dict['target_rmsd'] = ros.core.scoring.CA_rmsd(in_model_pose, out_model_pose, monomer_size+1, in_model_pose.size())
  
  target_map = ros.std.map_core_id_AtomID_core_id_AtomID()
  for seqpos in range(monomer_size+1, in_model_pose.size()+1):
    target_map[ros.core.id.AtomID(2, seqpos)] = ros.core.id.AtomID(2, seqpos) # CA
  ros.core.scoring.superimpose_pose(out_pose, in_model_pose, target_map)

  binder_map = ros.std.map_core_id_AtomID_core_id_AtomID()
  for seqpos in range(1, monomer_size+1):
    binder_map[ros.core.id.AtomID(2, seqpos)] = ros.core.id.AtomID(2, seqpos) # CA
  score_dict['interface_rmsd'] = ros.core.scoring.rms_at_corresponding_atoms_no_super(out_pose, in_model_pose, binder_map)

  score_dict['time'] = int(round(time.time() - start_time))
  score_dict['seed_idx'] = seed_idx

  print("min_pae: %.2f pae_interaction: %.2f interface_rmsd: %.1f binder_rmsd: %.1f target_rmsd: %.1f time: %i"%(score_dict['min_pae_interaction'], 
      score_dict['pae_interaction'], score_dict['interface_rmsd'], score_dict['binder_rmsd'], score_dict['target_rmsd'], score_dict['time']))

  return score_dict, out_pose


def write_ppi_result(result, other_inputs, tag=None, seed_idx='None', start_time=0, write=True):
  if tag is None:
    tag = other_inputs['tag'] + '_af3pred'
    
  processed_result = post_processing.post_process_inference_result(result)
  out_model_pose = mmcif_to_pose(processed_result.cif.decode('ascii'))

  in_model_pose = other_inputs['model_pose']
  assert out_model_pose.sequence() == in_model_pose.sequence()

  score_dict, out_pose = calculate_ppi_scores(processed_result, in_model_pose, out_model_pose, seed_idx=seed_idx, start_time=start_time)

  # tag = other_inputs['tag'] + '_af3pred'

  out_pose.pdb_info(other_inputs['input_pose'].pdb_info())

  if write:

    if other_inputs['output_type'] == 'silent':
      silent_path = other_inputs['silent_file']
      sfd_out = ros.core.io.silent.SilentFileData( silent_path, False, False, "binary", ros.core.io.silent.SilentFileOptions())
      struct = sfd_out.create_SilentStructOP()
      struct.fill_struct( out_pose, tag )
      for scorename, value in score_dict.items():
        if isinstance(value, str):
          struct.add_string_value(scorename, value)
        else:
          struct.add_energy(scorename, value)
      sfd_out.add_structure(struct)
      sfd_out.write_silent_struct( struct, silent_path)
    else:
      out_path = os.path.join(other_inputs['output_dir'] , tag + '.pdb')
      out_pose.dump_pdb(out_path)

    scorefile_path = other_inputs['score_file']
    write_header = not os.path.exists(scorefile_path)
    add2scorefile(tag, scorefile_path, write_header=write_header, score_dict=score_dict)

  return score_dict['target_rmsd'] * -0.5


### bcov custom functions ##


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    print_all_paired_species: bool = False,
    other_inputs: dict = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    print_all_paired_species: If True, prints all species that are paired. 
      Note that this may slightly exceed the actual set of paired species in the
      featurized MSA, since this will be cropped to max_paired_sequences.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  start_time = time.time()
  print(f'\nRunning fold job {fold_input.name}...')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  # if os.path.exists(output_dir) and os.listdir(output_dir):
  #   new_output_dir = (
  #       f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
  #   )
  #   print(
  #       f'Output will be written in {new_output_dir} since {output_dir} is'
  #       ' non-empty.'
  #   )
  #   output_dir = new_output_dir
  # else:
  # print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  # write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        conformer_max_iterations=conformer_max_iterations,
        print_all_paired_species=print_all_paired_species,
    )
    # print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
        other_inputs=other_inputs,
        start_time=start_time
    )
    output = all_inference_results

  print(f'Fold job {fold_input.name} done\n') #, output written to {output_dir}\n')
  return output


def main(_):
  # Validate required paths
  if MODEL_DIR.value is None:
    raise ValueError(
        '--model_dir must be specified or AF3_MODEL_DIR environment variable must be set. '
        'Example: --model_dir=/path/to/models or export AF3_MODEL_DIR=/path/to/models'
    )
  if not os.path.exists(MODEL_DIR.value):
    raise FileNotFoundError(
        f'Model directory not found: {MODEL_DIR.value}. '
        'Please check --model_dir or AF3_MODEL_DIR environment variable.'
    )
  
  if DB_DIR.value is None or len(DB_DIR.value) == 0:
    raise ValueError(
        '--db_dir must be specified or AF3_DB_DIR environment variable must be set. '
        'Example: --db_dir=/path/to/databases or export AF3_DB_DIR=/path/to/databases'
    )
  for db_dir in DB_DIR.value:
    if not os.path.exists(db_dir):
      raise FileNotFoundError(
          f'Database directory not found: {db_dir}. '
          'Please check --db_dir or AF3_DB_DIR environment variable.'
      )
  
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )

  # if _JSON_PATH.value is None == _INPUT_DIR.value is None == _FASTA.value is None:
  #   raise ValueError(
  #       'Exactly one of --json_path or --input_dir or --fasta must be specified.'
  #   )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  # if _INPUT_DIR.value is not None:
  #   fold_inputs = folding_input.load_fold_inputs_from_dir(
  #       pathlib.Path(_INPUT_DIR.value)
  #   )
  # elif _JSON_PATH.value is not None:
  #   _jsonf = json.load( open( pathlib.Path(_JSON_PATH.value), "r") )
  #   if isinstance(_jsonf, dict) or (isinstance(_jsonf, list) and _jsonf[0]["dialect"] == "alphafoldserver"):
  #       # Default AF3 inputs
  #       fold_inputs = folding_input.load_fold_inputs_from_path(
  #           pathlib.Path(_JSON_PATH.value)
  #       )
  #   else:
  #       # IK: accepting multi-input JSON in alphafold3 dialect
  #       fold_inputs = load_fold_inputs_from_list(_jsonf)

  # elif _FASTA is not None:
  #   # IK: FASTA file input and ligands from commandline
  #   input_list = parse_fasta_input(_FASTA, _LIGANDS, _SEEDS)
  #   fold_inputs = load_fold_inputs_from_list(input_list)

    # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise



  if _SILENT_PATH.value is not None or _PDBS.value is not None:
    expanded_fold_inputs = load_ppi_inputs(silent_path=_SILENT_PATH.value, pdbs=_PDBS.value, output_dir=_OUTPUT_DIR.value)

  else:
    raise AssertionError(
        'Exactly one of --silent or --pdbs must be specified.'
    )

  if expanded_fold_inputs is None:
    print("All jobs finished")
    return


  if _BUCKETS.value is not None:
    buckets = _BUCKETS.value
  else:
    max_bucket_size = 5120
    buckets = np.arange(_BUCKET_STEP.value, max_bucket_size, _BUCKET_STEP.value, dtype=int)


  if _RUN_INFERENCE.value:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      compute_capability = float(
          gpu_devices[_GPU_DEVICE.value].compute_capability
      )
      if compute_capability < 6.0:
        raise ValueError(
            'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
            ' https://developer.nvidia.com/cuda-gpus).'
        )
      elif 7.0 <= compute_capability < 8.0:
        xla_flags = os.environ.get('XLA_FLAGS')
        required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
        if not xla_flags or required_flag not in xla_flags:
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
              f' include "{required_flag}".'
          )
        if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the'
              ' --flash_attention_implementation must be set to "xla".'
          )

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n' + '\n'.join(notice) + '\n')

  if _RUN_DATA_PIPELINE.value:
    # Validate binary paths if data pipeline is enabled
    required_binaries = {
        'jackhmmer': _JACKHMMER_BINARY_PATH.value,
        'nhmmer': _NHMMER_BINARY_PATH.value,
        'hmmalign': _HMMALIGN_BINARY_PATH.value,
        'hmmsearch': _HMMSEARCH_BINARY_PATH.value,
        'hmmbuild': _HMMBUILD_BINARY_PATH.value,
    }
    missing_binaries = []
    for name, path in required_binaries.items():
      if not path or not os.path.exists(path):
        missing_binaries.append(f"{name} (path: {path or 'not found'})")
    if missing_binaries:
      raise FileNotFoundError(
          f'Required MSA binaries not found: {", ".join(missing_binaries)}. '
          'Please install them or set the corresponding environment variables '
          '(e.g., JACKHMMER_BINARY_PATH, NHMMER_BINARY_PATH, etc.)'
      )
    
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
        small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
        mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
        uniprot_cluster_annot_database_path=expand_path(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
        ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
        rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
        rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
        max_template_date=max_template_date,
    )
  else:
    data_pipeline_config = None

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend='gpu')
    print(
        f'Found local devices: {devices}, using device {_GPU_DEVICE.value}:'
        f' {devices[_GPU_DEVICE.value]}'
    )

    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
            ),
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=devices[_GPU_DEVICE.value],
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
  else:
    model_runner = None

  num_fold_inputs = 0
  other_inputs = None
  for other_inputs, fold_input in expanded_fold_inputs:
    if _NUM_SEEDS.value is not None:
      print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
      fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
        buckets=tuple(int(bucket) for bucket in buckets),
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        print_all_paired_species=_PRINT_ALL_PAIRED_SPECIES.value,
        other_inputs=other_inputs
    )
    num_fold_inputs += 1

  print(f'Done running {num_fold_inputs} fold jobs.')

  if other_inputs is not None:
    with open(other_inputs['checkpoint'], 'a') as f:
      f.write('DONE\n')


if __name__ == '__main__':
  flags.mark_flags_as_required([])
  app.run(main)
