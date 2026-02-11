import torch
from pathlib import Path
from ml.pc_remover import remove_path_condition_root
import shutil

def main():

    source_path = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\dir1")
    output_path = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\DIR")
        
    output_path.mkdir(parents=True, exist_ok=True)

    all_paths = list(source_path.rglob("*"))
        
    for item in all_paths:
        rel_path = item.relative_to(source_path)
        output_item_path = output_path / rel_path
        
        if item.is_dir():
            output_item_path.mkdir(parents=True, exist_ok=True)
        else:
            output_item_path.parent.mkdir(parents=True, exist_ok=True)
            
            if item.suffix == '.pt':
                data = torch.load(item, map_location='cpu', weights_only=False)
                processed_data = remove_path_condition_root(data)
                torch.save(processed_data, output_item_path)
            else:
                shutil.copy2(item, output_item_path)

if __name__ == "__main__":
    main()