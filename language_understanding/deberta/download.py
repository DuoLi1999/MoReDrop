from transformers import DebertaV2Model, DebertaV2Tokenizer

# 下载RoBERTa模型和tokenizer
model_name = 'microsoft/deberta-v3-xsmall'
model = DebertaV2Model.from_pretrained(model_name)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

# 将模型和tokenizer保存到本地文件夹
output_dir = 'ckp'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)