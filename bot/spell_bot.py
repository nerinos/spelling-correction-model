import telebot
import torch
from transformer import Encoder, Decoder, Transformer
from torchtext.legacy.data import Field
from pytorch_transformers import BertTokenizer
import pickle
from datetime import datetime
import nltk


def correct_sentence(sentence, src_field, trg_field, model, device, max_len=80):
    model.eval()

    if isinstance(sentence, str):
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        tokens = [token for token in tokenizer.tokenize(sentence)]
    else:
        tokens = [token for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def beautify(text_lst):
    symbols = set([",", ".", "?", "!", ":"])
    result = ""

    for item in text_lst:
        if item.startswith("##"):
            result += item[2:]
        elif item in symbols:
            result += item
        else:
            result += " " + item
    return result


def try_correct(sentence):
    translation, attention = correct_sentence(sentence, SRC, TRG, model, device)
    return beautify(translation[:-1])


def create_transformer(SRC, TRG):
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                HID_DIM,
                ENC_LAYERS,
                ENC_HEADS,
                ENC_PF_DIM,
                ENC_DROPOUT,
                device)

    dec = Decoder(OUTPUT_DIM,
                HID_DIM,
                DEC_LAYERS,
                DEC_HEADS,
                DEC_PF_DIM,
                DEC_DROPOUT,
                device)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    return model

# Создаем экземпляр бота
bot = telebot.TeleBot('1867864497:AAEPjEOmatknyq6n-pbYLI73E-cQwMTPAjI')

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
SRC = Field(tokenize=tokenizer.tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            fix_length=80,
            lower=False,
            batch_first=True)

TRG = Field(tokenize=tokenizer.tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            fix_length=80,
            lower=False,
            batch_first=True)
model = None
fields = {'trg': TRG, 'src': SRC}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Функция, обрабатывающая команду /start
@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, f"""
    Привет!, {m.chat.username}
Я - бот, который исправляет ошибки.
Учти, что мой словарь специфичен, ведь я читал только художественную литературу:)
Просто напиши мне сообщение, а я попробую что-нибудь в нём исправить!
    """)


# Получение сообщений от юзера
@bot.message_handler(content_types=["text"])
def handle_text(message):
    sent = message.text
    sent_text = nltk.sent_tokenize(sent)  # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    corrected = ""
    for sentence in sent_text:
        corrected += try_correct(sentence)
    corrected = corrected.strip()
    with open("bot_log.txt", "a") as f:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        input = message.text.replace("\n", " ")
        output = corrected.replace("\n", " ")
        log_string = f"{dt_string}/{message.chat.username}({message.chat.first_name} {message.chat.last_name}) wrote: [{input}] corrected: [{output}]\n"
        print(log_string)
        f.write(log_string)
    bot.send_message(message.chat.id, corrected)



if __name__=="__main__":
    with open("SRC_transformer_best.pickle", "rb") as f:
        SRC.vocab = pickle.load(f)
    with open("TRG_transformer_best.pickle", "rb") as f:
        TRG.vocab = pickle.load(f)
    nltk.download('punkt')
    model = create_transformer(SRC, TRG)
    model.load_state_dict(torch.load('transformer_best_params.pth', map_location=torch.device('cpu')))
    print("Starting bot...")
    bot.polling(none_stop=True, interval=0)



