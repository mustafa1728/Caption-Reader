import pandas as pd
import torch

def get_caption_transforms(vocab_file_path):

    df = pd.read_csv(vocab_file_path)
    vocab = df.iloc[0, :].values
    word_to_idx = {vocab[i]: i for i in range(len(vocab))}
    # idx_to_word = {vocab[i]: i for i in range(len(vocab))}

    def tokenize(caption):
        word_list =  caption.split(" ")
        return [w.lower() for w in word_list]

    def add_start_end(word_list):
        return ["<start>"] + word_list + ["<end>"]

    def get_indices(word_list):
        return [word_to_idx[w] if w in vocab else word_to_idx["<unk>"] for w in word_list]

    def composed_transform(caption):
        return get_indices(add_start_end(tokenize(caption)))

    return composed_transform

def get_img_transforms(output_size):

    def pad_resize(image):
        out_img = torch.zeros([3]+output_size)

        img_size = image.size()

        if img_size[1]/output_size[1] > img_size[2]/output_size[2]:
            resized_size = (3, output_size[1], int(img_size[2] * output_size[1]/img_size[1]))
        else:
            resized_size = (3, int(output_size[1] * output_size[2]/img_size[2]), output_size[2])
        
        resized_img = image.view(resized_size)
        start_id1 = output_size[1]//2 - resized_size[1]//2
        start_id2 = output_size[2]//2 - resized_size[2]//2
        out_img[:, start_id1:start_id1+resized_size[1], start_id2:resized_size[2]] = resized_img

        return out_img

    return pad_resize