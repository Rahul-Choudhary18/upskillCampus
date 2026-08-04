import os
import glob
import json

class DatasetLoader:
    def __init__(self, dataset_dir=os.path.join('dataset', 'agri_data', 'data'), classes_file=os.path.join('dataset', 'classes.txt')):
        self.dataset_dir = dataset_dir
        self.classes_file = classes_file
        self.classes = ['crop', 'weed']
        self.load_classes()
        
    def load_classes(self):
        if os.path.exists(self.classes_file):
            with open(self.classes_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    self.classes = lines

    def get_stats(self):
        images = glob.glob(os.path.join(self.dataset_dir, '*.jpeg')) + glob.glob(os.path.join(self.dataset_dir, '*.jpg'))
        txts = glob.glob(os.path.join(self.dataset_dir, '*.txt'))
        
        crop_count = 0
        weed_count = 0
        total_boxes = 0
        
        for txt in txts:
            with open(txt, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        if cid == 0:
                            crop_count += 1
                        elif cid == 1:
                            weed_count += 1
                        total_boxes += 1
                        
        return {
            'total_images': len(images),
            'total_annotation_files': len(txts),
            'total_bounding_boxes': total_boxes,
            'class_distribution': {
                'crop': {'count': crop_count, 'label': self.classes[0] if len(self.classes) > 0 else 'crop'},
                'weed': {'count': weed_count, 'label': self.classes[1] if len(self.classes) > 1 else 'weed'}
            },
            'pipeline_steps': [
                {
                    'step': 1,
                    'title': 'Photo Collection',
                    'description': 'Captured raw high-resolution (4000x3000) field photos of sesame crops and competing weeds.',
                    'stat': '589 Raw Photos'
                },
                {
                    'step': 2,
                    'title': 'Data Cleaning & Quality Audit',
                    'description': 'Removed blurry, overexposed, and low-visibility frames to prevent model noise.',
                    'stat': '546 Cleaned Photos'
                },
                {
                    'step': 3,
                    'title': 'Dimension Standardization',
                    'description': 'Resized high-res 4000x3000 images to 512x512x3 color dimensions for model efficiency.',
                    'stat': '512 x 512 Color Images'
                },
                {
                    'step': 4,
                    'title': 'Keras Data Augmentation',
                    'description': 'Expanded 546 images into 1,300 using rotation, zoom, flips, and shear augmentation.',
                    'stat': '1,300 Augmented Images'
                },
                {
                    'step': 5,
                    'title': 'YOLO Bounding Box Labeling',
                    'description': 'Manually labeled 2,072 precise bounding boxes (Class 0: Sesame Crop, Class 1: Weed).',
                    'stat': '2,072 Object Annotations'
                }
            ]
        }

    def get_sample_list(self, limit=30):
        images = glob.glob(os.path.join(self.dataset_dir, '*.jpeg')) + glob.glob(os.path.join(self.dataset_dir, '*.jpg'))
        sample_list = []
        
        for img_path in sorted(images)[:limit]:
            filename = os.path.basename(img_path)
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            
            crops = 0
            weeds = 0
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid = int(parts[0])
                            if cid == 0: crops += 1
                            elif cid == 1: weeds += 1
                            
            sample_list.append({
                'filename': filename,
                'crops': crops,
                'weeds': weeds,
                'total_objects': crops + weeds
            })
            
        return sample_list

if __name__ == '__main__':
    loader = DatasetLoader()
    stats = loader.get_stats()
    print("Dataset Stats:", json.dumps(stats, indent=2))
