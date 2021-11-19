from models.capnet import CapNet
from data.dataset import CaptionDataset, generate_vocabulary
from data.dataloader import CaptionsDataLoader
from data.transforms import get_img_transforms, get_caption_transforms

import os
import torch
from tqdm import tqdm
import argparse
import logging

USE_GPU = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description='Caption Reader Training')
    parser.add_argument('--root', type=str, help='path to dataset images root directory', default="../", required=False)
    parser.add_argument('--ann', type=str, help='path to annotation file', default="../Train_text.tsv", required=False)
    parser.add_argument('--vocab_path', type=str, help='path to save vocabulary', default="../vocabulary.csv", required=False)
    parser.add_argument('--bs', type=int, help='batch size', default=32, required=False)
    parser.add_argument('--epoch', type=int, help='number of epochs to train', default=10, required=False)
    parser.add_argument('--ckpt_path', type=str, help='path to save checkpoints', default="./checkpoints", required=False)
    parser.add_argument('--seed', type=int, help='seed', default=0, required=False)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    vocab_path = args.vocab_path
    vocab = generate_vocabulary(args.ann, vocab_path)
    train_dataset = CaptionDataset(
        img_prefix=args.root,
        ann_file=args.ann,
        img_transforms=get_img_transforms(output_size=(256, 256)),
        cap_transforms=get_caption_transforms(vocab_file_path=vocab_path),
        vocab_path=vocab_path,
    )
    train_dataloader = CaptionsDataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        seed=args.seed,
    )
    model = CapNet(
        embedding_dim=128,
        lstm_size=256,
        vocab_size=len(vocab),
        use_gpu=USE_GPU,
    )
    if USE_GPU:
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    os.makedirs(args.ckpt_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.ckpt_path, 'train.log'), filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)
    logging.info("Start new training")
    for epoch in range(1, args.epoch+1):
        for batch in tqdm(train_dataloader):
            images, captions = batch
            padded_caps = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)
            packed_caps = torch.nn.utils.rnn.pack_padded_sequence(padded_caps, batch_first=True, lengths=[cap.size(0) for cap in captions], enforce_sorted=False)
            batch_images = torch.stack(images)

            if USE_GPU:
                batch_images = batch_images.cuda()
                packed_caps = packed_caps.cuda()
            pred_captions = model(batch_images, packed_caps)

            loss = criterion(pred_captions, packed_caps.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        to_print_str = "Epoch {}/{} | Loss: {}".format(epoch, args.epoch, loss.item())
        print(to_print_str)
        logging.info(to_print_str)
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, "model_{}.pth".format(epoch)))

if __name__ == "__main__":
    main()