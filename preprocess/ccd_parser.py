import os
import json
import gemmi
from typing import Optional, Dict

class LocalCcdParser:
    """
    一个用于从本地 components.cif 文件中解析化学成分信息的类。

    该类会自动为 components.cif 文件创建并缓存一个索引，以实现快速查找。
    首次实例化时，如果索引文件不存在，会花费一些时间来构建索引。
    后续实例化将直接加载缓存的索引，速度非常快。
    """
    def __init__(self, cif_path: str):
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"指定的 CIF 文件不存在: {cif_path}")
        
        self.cif_path = cif_path
        self.index_path = f"{cif_path}.index.json"
        self.index: Dict[str, int] = self._load_or_build_index()

    def _build_index(self) -> Dict[str, int]:
        """
        (此函数无需修改，本身就是正确的)
        遍历 CIF 文件，构建 CCD code 到其在文件中字节偏移量的索引。
        """
        print(f"索引文件 '{self.index_path}' 不存在，正在构建索引... (这可能需要一两分钟)")
        index = {}
        with open(self.cif_path, 'rb') as f:
            offset = 0
            for line_bytes in f:
                # 仅在需要检查内容时解码
                if line_bytes.startswith(b'data_'):
                    line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                    code = line_str[5:]
                    index[code] = offset
                offset = f.tell()
        
        with open(self.index_path, 'w') as f_out:
            json.dump(index, f_out)
        
        print("索引构建完成并已保存。")
        return index

    def _load_or_build_index(self) -> Dict[str, int]:
        if os.path.exists(self.index_path):
            print(f"正在从 '{self.index_path}' 加载已缓存的索引...")
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return self._build_index()

    def get_smiles(self, ccd_code: str) -> Optional[str]:
        """
        (*** 已修正 ***)
        从本地 CIF 文件中获取给定 CCD code 的 SMILES 字符串。
        """
        code = ccd_code.upper()
        
        if code not in self.index:
            print(f"错误：在索引中未找到 CCD code '{code}'。")
            return None

        start_offset = self.index[code]
        
        # *** 修正点 1: 必须使用二进制模式 'rb' 打开文件 ***
        with open(self.cif_path, 'rb') as f:
            f.seek(start_offset)
            
            # 读取该 code 对应的数据块（字节串形式）
            cif_block_bytes_list = []
            for line_bytes in f:
                # *** 修正点 2: 对字节串进行判断 ***
                # 遇到下一个数据块的开头就停止
                if line_bytes.startswith(b'data_') and cif_block_bytes_list:
                    break
                cif_block_bytes_list.append(line_bytes)
            
            # 将字节串列表连接起来，然后一次性解码为字符串
            cif_block_bstring = b"".join(cif_block_bytes_list)
            cif_block_str = cif_block_bstring.decode('utf-8', errors='ignore')

        # 使用 Gemmi 解析这个小的数据块字符串
        try:
            doc = gemmi.cif.read_string(cif_block_str)
            block = doc.sole_block()
            
            smiles = block.find_loop('_pdbx_chem_comp_descriptor.descriptor')
            return smiles

        except Exception as e:
            print(f"解析 CCD code '{code}' 时出错: {e}")
            return None