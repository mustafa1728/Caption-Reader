from models.capnet import CapNet
from data.dataset import CaptionDataset, generate_vocabulary
from data.dataloader import CaptionsDataLoader
from data.transforms import get_img_transforms, get_caption_transforms

import os
import torch
from tqdm import tqdm
import logging
logging.basicConfig(filename='train.log', filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)

USE_GPU = torch.cuda.is_available()

def main():
    vocab_path = "../vocabulary.csv"
    vocab = generate_vocabulary("../Train_text.tsv", vocab_path)
    train_dataset = CaptionDataset(
        img_prefix="../",
        ann_file="../Train_text.tsv",
        img_transforms=get_img_transforms(output_size=(256, 256)),
        cap_transforms=get_caption_transforms(vocab_file_path=vocab_path),
        vocab_path=vocab_path,
    )
    train_dataloader = CaptionsDataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        seed=0,
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
    no_epochs = 10

    os.makedirs("./checkpoints", exist_ok=True)
    logging.info("Start new training")
    for epoch in range(1, no_epochs+1):
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
        to_print_str = "Epoch {}/{} | Loss: {}".format(epoch, no_epochs, loss.item())
        print(to_print_str)
        logging.info(to_print_str)
        torch.save(model.state_dict(), "./checkpoints/model_{}.pth".format(epoch))

if __name__ == "__main__":
    main()