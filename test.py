import torch.nn.functional as F
import transformers
import torch
max_history_len = 1
max_len = 25
temperature = 1
repetition_penalty = 1
topk = 8
topp = 0
cuda = False

tokenizer = torch.load("manmade/tokenizer.ckpt", map_location=torch.device('cpu'))

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def main():
    #args = set_args()
    #logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    #args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if cuda else 'cpu'
    #logger.info('using device:{}'.format(device))
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = torch.load("manmade/tokenizer.ckpt", map_location=torch.device('cpu'))
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained("./model_1/")  # load pretrained model
    model = model.to(device)
    model.eval()
    #if args.save_samples_path:
    #    if not os.path.exists(args.save_samples_path):
    #        os.makedirs(args.save_samples_path)
    #    samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
    #    samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            # text = "你好"
            #if args.save_samples_path:
            #    samples_file.write("user:{}\n".format(text))
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.convert_tokens_to_ids('[MASK]')]  # 每个input以[MASK]为开头
            for history_id, history_utr in enumerate(history[-max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)
            response = []  # 根据context，生成的response
            # 最多生成max_len个token
            for _ in range(max_len):
                outputs = model(input_ids=input_ids)
                #print(input_ids)
                #logits = outputs
                next_token_logits = outputs[0][0, -1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(response):
                    next_token_logits[id] /= repetition_penalty
                next_token_logits = next_token_logits / temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.cls_token_id:  # 遇到[cls]则表明response生成结束
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot:" + "".join(text).replace("#", ""))
            #if args.save_samples_path:
            #    samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            #if args.save_samples_path:
            #    samples_file.close()
            break


main()