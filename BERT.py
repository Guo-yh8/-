from transformers import BertForQuestionAnswering

def get_model():
    model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')
    return model
