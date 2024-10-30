from matplotlib import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import clip
from sklearn.decomposition import PCA

class EEGDataset_360EEG(Dataset):
    def __init__(self, data_path, images_path, split_file, train=True, transform=None):
        """
        Args:
            data_path (str): Path to the EEG data file (e.g., eeg_data.npy).
            images_path (str): Path to the folder containing images.
            split_file (str): Path to the train/test split file (e.g., train_test_idx.npy).
            train (bool): If True, loads the training set. Otherwise, loads the test set.
            transform (callable, optional): Optional transform to be applied on the images.
        """
        self.data_path = data_path
        self.images_path = images_path
        self.split_file = split_file
        self.train = train
        self.transform = transform

        # Load the EEG data
        self.data_dict = np.load(self.data_path, allow_pickle=True).item()
        self.eeg_data = self.data_dict['eeg']
        self.labels = self.data_dict['label']
        # self.image_indices = self.data_dict['image']
        self.image_indices = [f"{i:03}" for i in range(1, 241)]
        self.label_map = self.data_dict['label_map']
        self.electrodes = self.data_dict['electrodes']
        self.label_map_reverse = {v: k for k, v in self.label_map.items()}

        # Load the train/test split indices
        split_dict = np.load(self.split_file, allow_pickle=True).item()
        self.indices = split_dict['train_idx'] if train else split_dict['test_idx']
        # Load models and processors, move to CPU
        self.clip_model, self.processor_clip = clip.load("ViT-L/14", device="cuda", download_root='./pretrained/clip')

        for param in self.clip_model.parameters():
            param.requires_grad = False


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the correct index for this dataset split
        index = self.indices[idx]

        # Get the corresponding EEG segment and label
        eeg_segment = self.eeg_data[index].T  # Shape: (electrodes, sampling points)
        label = self.labels[index]
        label_text = self.label_map_reverse[label]
        
        # Get the image corresponding to this EEG segment
        image_idx = self.image_indices[index]
        image_path = os.path.join(self.images_path, f"{image_idx}.png")  # Assuming images are stored as .jpg
        image = Image.open(image_path)

        # Apply any image transformations (if provided)
        if self.transform:
            image_raw = self.transform(image)

        # Preprocess the EEG segment
        eeg_segment = self.preprocess_eeg(eeg_segment)

        # Convert EEG data and label to torch tensors
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        image_processed_clip = self.processor_clip(image).unsqueeze(0).to("cuda")
        image_embedding_clip = self.clip_model.encode_image(image_processed_clip).squeeze(0)

        return {"eeg":eeg_tensor, "images":image_raw, "labels":label_tensor, "image_embedding":image_embedding_clip, "labels_text":label_text}

    def preprocess_eeg(self, eeg_segment):
        """
        Preprocesses the EEG data by applying Min-Max normalization and a simple denoising filter.
        Args:
            eeg_segment (np.array): EEG data with shape (electrodes, sampling points).
        Returns:
            np.array: Preprocessed EEG data.
        """
        # Apply a simple denoising filter (Butterworth low-pass filter)
        eeg_segment = self.denoise_eeg(eeg_segment)

        # Min-Max normalization (for each electrode independently)
        eeg_min = np.min(eeg_segment, axis=1, keepdims=True)
        eeg_max = np.max(eeg_segment, axis=1, keepdims=True)
        eeg_segment = (eeg_segment - eeg_min) / (eeg_max - eeg_min + 1e-8)  # Avoid division by zero

        # Step 3: Apply PCA for dimensionality reduction
        pca = PCA(n_components= 4)
        eeg_segment = pca.fit_transform(eeg_segment.T).T  # Transpose for PCA, then back

        return eeg_segment

    def denoise_eeg(self, eeg_segment, lowcut=4, highcut=47, fs=128, order=5):
        """
        Applies a Butterworth bandpass filter to remove noise from the EEG signal.
        Args:
            eeg_segment (np.array): EEG data with shape (electrodes, sampling points).
            lowcut (float): Low cut frequency for the bandpass filter.
            highcut (float): High cut frequency for the bandpass filter.
            fs (float): Sampling frequency (default is 250 Hz).
            order (int): Order of the Butterworth filter.
        Returns:
            np.array: Denoised EEG data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        denoised_data = filtfilt(b, a, eeg_segment, axis=-1)  # Apply along the last axis (sampling points)
        return denoised_data
    

class EEGDataset_ImageNet(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=None, subject=0):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path, map_location=torch.device('cpu'))
        
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '/root/autodl-tmp/LGM/dreamdiffusion/datasets/imageNet_images'
        self.transform = image_transform if image_transform else transforms.Compose([
            transforms.ToTensor(),  # 确保图像被转换为 Tensor
            transforms.Resize((256, 256)),  # 可选：调整图像大小
            # transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
        ])
        self.num_voxels = 440
        self.data_len = 512
        # Compute size
        self.size = len(self.data)

        
        # Load models and processors, move to CPU
        self.clip_model, self.processor_clip = clip.load("ViT-L/14", device="cuda", download_root='./pretrained/clip')

        for param in self.clip_model.parameters():
            param.requires_grad = False

        

    # Helper function to extract the label from the filename
    def extract_label_from_filename(self, filename):
        """
        Extracts the label string from the filename.
        Assumes the label is included as part of the filename.
        Example: "dog_1234.jpg" -> "dog"
        """
        # Assuming the label is before the first underscore (e.g., "dog_1234.jpg" -> "dog")
        label_str = filename.split('_')[0]
        return label_str

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460, :]
        
        # Preprocess EEG data
        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        
        # Get label
        label = torch.tensor(self.data[i]["label"])
        
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name + '.JPEG')
        image_raw = Image.open(image_path).convert('RGB') 

        # Get labels_text from image_name
        class_code = image_name.split('_')[0]  # 提取类别代码
        labels_text = image_net_dict().propmt_dict.get(class_code, "Unknown")  # 查找对应的标签文本，如果没有则返回 'Unknown'
        
        image_processed_clip = self.processor_clip(image_raw).unsqueeze(0).to("cuda")
        image_embedding_clip = self.clip_model.encode_image(image_processed_clip).squeeze(0)

        # inputs_blip = self.processor_blip(image_raw, return_tensors="pt").to("cpu")
        # generated_ids = self.image2text_model.generate(**inputs_blip, max_length=30)
        # generated_text = self.processor_blip.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # token_prompts = self.text_tokenizer(generated_text, return_tensors='pt', padding="max_length", max_length=77, truncation=True).to("cuda")
        # latent_prompts = self.text_encoder(**token_prompts, return_dict=False)[0].squeeze(0)
        
        # image = np.array(image_raw) / 255.0
        
        # return {'eeg': eeg, 'image': self.image_transform(image), 'image_embedding': image_embedding_clip, 'text': generated_text, 'latent_prompts': latent_prompts}
        # return {'eeg': eeg, 'label':label, 'image': self.image_transform(image), 'image_embedding': image_embedding_clip}
        return {"eeg":eeg, "labels":label, "images":self.transform(image_raw), "labels_text":labels_text, "image_embedding":image_embedding_clip}
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=0):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../dreamdiffusion/datasets/eeg_5_95_std.pth', 
            splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',
            # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',
            image_transform=None, subject = 0):
    # if subject == 0:
        # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth'
    if isinstance(image_transform, list):
        dataset_train = EEGDataset_ImageNet(eeg_signals_path, image_transform[0], subject )
        dataset_test = EEGDataset_ImageNet(eeg_signals_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset_ImageNet(eeg_signals_path, image_transform, subject)
        dataset_test = EEGDataset_ImageNet(eeg_signals_path, image_transform, subject)
    split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train', subject= subject)
    split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test', subject = subject)
    return (split_train, split_test)

class image_net_dict:
    def __init__(self):
        self.propmt_dict = {'n02106662': 'german shepherd dog',
            'n02124075': 'cat ',
            'n02281787': 'lycaenid butterfly',
            'n02389026': 'sorrel horse',
            'n02492035': 'Cebus capucinus',
            'n02504458': 'African elephant',
            'n02510455': 'panda',
            'n02607072': 'anemone fish',
            'n02690373': 'airliner',
            'n02906734': 'broom',
            'n02951358': 'canoe or kayak',
            'n02992529': 'cellular telephone',
            'n03063599': 'coffee mug',
            'n03100240': 'old convertible',
            'n03180011': 'desktop computer',
            'n03197337': 'digital watch',
            'n03272010': 'electric guitar',
            'n03272562': 'electric locomotive',
            'n03297495': 'espresso maker',
            'n03376595': 'folding chair',
            'n03445777': 'golf ball',
            'n03452741': 'grand piano',
            'n03584829': 'smoothing iron',
            'n03590841': 'Orange jack-o’-lantern',
            'n03709823': 'mailbag',
            'n03773504': 'missile',
            'n03775071': 'mitten,glove',
            'n03792782': 'mountain bike, all-terrain bike',
            'n03792972': 'mountain tent',
            'n03877472': 'pajama',
            'n03888257': 'parachute',
            'n03982430': 'pool table, billiard table, snooker table ',
            'n04044716': 'radio telescope',
            'n04069434': 'eflex camera',
            'n04086273': 'revolver, six-shooter',
            'n04120489': 'running shoe',
            'n07753592': 'banana',
            'n07873807': 'pizza',
            'n11939491': 'daisy',
            'n13054560': 'bolete'
            }
        self.label_number_dict={
            '[12]': 'n02106662',
            '[39]': 'n02124075',
            '[11]': 'n02281787',
            '[0]': 'n02389026',
            '[21]': 'n02492035',
            '[35]': 'n02504458',
            '[8]': 'n02510455',
            '[3]': 'n02607072',
            '[36]': 'n02690373',
            '[18]': 'n02906734',
            '[10]': 'n02951358',
            '[15]': 'n02992529',
            '[5]': 'n03063599',
            '[24]': 'n03100240',
            '[17]': 'n03180011',
            '[34]': 'n03197337',
            '[28]': 'n03272010',
            '[37]': 'n03272562',
            '[4]': 'n03297495',
            '[25]': 'n03376595',
            '[16]': 'n03445777',
            '[30]': 'n03452741',
            '[2]': 'n03584829',
            '[14]': 'n03590841',
            '[23]': 'n03709823',
            '[20]': 'n03773504',
            '[27]': 'n03775071',
            '[6]': 'n03792782',
            '[31]': 'n03792972',
            '[26]': 'n03877472',
            '[1]': 'n03888257',
            '[22]': 'n03982430',
            '[38]': 'n04044716',
            '[29]': 'n04069434',
            '[7]': 'n04086273',
            '[13]': 'n04120489',
            '[32]': 'n07753592',
            '[19]': 'n07873807',
            '[9]': 'n11939491',
            '[33]': 'n13054560'
            }
    

    