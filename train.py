from models.capnet import CapNet
from data.dataset import CaptionDataset, generate_vocabulary
from data.dataloader import CaptionsDataLoader
from data.transforms import get_img_transforms, get_caption_transforms

import torch
from tqdm import tqdm

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
        batch_size=8,
        shuffle=True,
        seed=0,
    )
    model = CapNet(
        embedding_dim=128,
        lstm_size=256,
        vocab_size=len(vocab),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    no_epochs = 10

    for epoch in range(1, no_epochs+1):
        for batch in tqdm(train_dataloader):
            images, captions = batch
            pred_captions = model(torch.stack(images), captions)
            loss = criterion(pred_captions.view(-1, pred_captions.size()[2]), captions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {}/{} | Loss: {}".format(epoch, no_epochs, loss.item()))

if __name__ == "__main__":
    main()