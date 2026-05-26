import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import torch.nn.functional as F


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, stage1_to_2
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

import runpy

import torch
import gc

# 清理显存
torch.cuda.empty_cache()
gc.collect()

# 设置 PyTorch 的显存分配策略
torch.cuda.set_per_process_memory_fraction(0.95)  # 使用 95% 的显存


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if "mask_index" in line:
            mask_index = torch.tensor(line["mask_index"])
        else:
            mask_index = None
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, mask_index

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, mask_index = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    if mask_index is None or all(m is None for m in mask_index):
        mask_index = None
    else:
        mask_index = torch.stack(mask_index, dim=0)
    return input_ids, image_tensors, image_sizes, mask_index


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    key_text_tokenized = torch.tensor(tokenizer('').input_ids[2:]).to(device='cuda', non_blocking=True)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.stage1_question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.stage1_answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    for (input_ids, image_tensor, image_sizes, _), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, logits, attn_map = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        # import pdb; pdb.set_trace()
        # Debug: 打印原始 attn_map 信息
        if idx == questions[0]["question_id"]:
            print(f"\n=== Stage 1 - Raw attn_map info ===")
            if isinstance(attn_map, (list, tuple)):
                print(f"attn_map is list/tuple with length: {len(attn_map)}")
                if len(attn_map) > 0:
                    print(f"First element shape: {attn_map[0].shape if hasattr(attn_map[0], 'shape') else 'N/A'}")
            else:
                print(f"attn_map shape: {attn_map.shape if hasattr(attn_map, 'shape') else 'N/A'}")        

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Compute confidence (variance over token probabilities for 'A' and 'B')
        logit = F.softmax(logits[0], dim=1)
        # Derive token ids for 'A' and 'B' robustly
        token_id_A = tokenizer('A', add_special_tokens=False).input_ids[0]
        token_id_B = tokenizer('B', add_special_tokens=False).input_ids[0]

        vars = torch.var(logit[:, [token_id_A, token_id_B]])

        last_step_probs = logit[-1]
        prob_A = last_step_probs[token_id_A].item()
        prob_B = last_step_probs[token_id_B].item()

        if vars.item() < args.beta_conf:
            pos_pct = int(round(prob_A * 100))
            neg_pct = int(round(prob_B * 100))
            outputs = f"{pos_pct} {neg_pct}"

        prompt_ids = []
        for row in input_ids:
            separator_index = (row == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
            output_row = row[separator_index + 1:]
            prompt_ids.append(output_row)
        prompt_ids = torch.stack(prompt_ids)

        rows, cols = prompt_ids.shape
        match_indices = []
        for row in range(rows):
            for i in range(cols - len(key_text_tokenized) + 1):
                if torch.equal(prompt_ids[row, i:i + len(key_text_tokenized)], key_text_tokenized):
                    match_indices.append(torch.tensor(list(range(i, i+len(key_text_tokenized)))))
        
        if idx == questions[0]["question_id"]:
            print(f"prompt_ids shape: {prompt_ids.shape}")
            print(f"key_text_tokenized length: {len(key_text_tokenized)}")
            print(f"Number of matches found: {len(match_indices)}")
            if match_indices:
                print(f"Match indices positions: {match_indices}")        



        if match_indices:
            match_indices_tensor = torch.stack(match_indices)

            attn_map_new = []
            for i,map in enumerate(attn_map):
                attn_map_new.append(map[match_indices_tensor[i]])
            attn_map = torch.stack(attn_map_new)

            # Debug: 打印 attn_map 的形状和统计信息
            if idx == questions[0]["question_id"]:  # 只对第一个样本打印
                print(f"\n=== Debug Info for first sample ===")
                print(f"Original attn_map shape after indexing: {attn_map.shape}")
            
            attn_map = torch.mean(attn_map, dim=1)
            
            if idx == questions[0]["question_id"]:
                print(f"After mean, attn_map shape: {attn_map.shape}")
                print(f"beta_mask: {args.beta_mask}")
                print(f"attn_map.shape[1]: {attn_map.shape[1]}")
            
            k = max(1, int(attn_map.shape[1] * args.beta_mask))
            
            if idx == questions[0]["question_id"]:
                print(f"k (number of tokens to mask): {k}")
                print(f"attn_map values (first 10): {attn_map[0, :min(10, attn_map.shape[1])]}")
                print(f"attn_map min: {attn_map.min().item()}, max: {attn_map.max().item()}")
            
            _, mask_index_tensor = torch.topk(-attn_map, k, dim=1)
            mask_index = mask_index_tensor[0].tolist()
            
            if idx == questions[0]["question_id"]:
                print(f"mask_index: {mask_index}")
                print(f"=================================\n")
        else:
            mask_index = None
            if idx == questions[0]["question_id"]:
                print(f"\n=== Debug Info ===")
                print(f"WARNING: No match found for 'Please focus on emotion' in prompt")
                print(f"=================================\n")


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "mask_index": mask_index,
                                   "model_id": model_name,
                                   "var": vars.item(),
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

    print('Stage1 to stage2...')
    stage1_to_2(args.stage1_answers_file, args.stage1_question_file, args.stage2_question_file)

        
    questions = [json.loads(q) for q in open(os.path.expanduser(args.stage2_question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.stage2_answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes, mask_index), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        current_mask_index = None
        if mask_index is not None:
             # 假设 batch size = 1
             if isinstance(mask_index, torch.Tensor):
                 current_mask_index = mask_index[0].tolist() if mask_index.dim() > 1 else mask_index.tolist()
             else:
                 current_mask_index = mask_index


        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, logits, _ = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                mask_flag=False, 
                mask_index=current_mask_index 

            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--stage1_question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--stage1_answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--stage2_question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--stage2_answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--beta_mask", type=float, default=0.1)
    parser.add_argument("--beta_conf", type=float, default=0.12)

# --- 在这里添加新的 efvr 参数 ---
    parser.add_argument("--use_efvr", action="store_true",
                        help="Flag to activate the merged efvr logic.")
    parser.add_argument("--efvr_entropy_threshold", type=float, default=0.7,
                        help="Entropy threshold to trigger efvr.")
    parser.add_argument("--efvr_retracing_ratio", type=float, default=0.3,
                        help="Ratio of visual information to re-inject.")
    parser.add_argument("--efvr_starting_layer", type=int, default=8,
                        help="efvr starting layer.")
    parser.add_argument("--efvr_ending_layer", type=int, default=15,
                        help="efvr ending layer.")
    parser.add_argument("--efvr_topk", type=int, default=10,
                        help="Top-k value for entropy calculation (Note: currently parsed but hard-coded in model).")
    # --- 结束添加 ---
    # --- MODIFICATION: 添加特定层列表 ---
    parser.add_argument("--efvr_trigger_layers", type=int, nargs='+', default=None,
                        help="A list of specific layer indices (e.g., 9 12 15) to *check* for entropy trigger. Overrides start/end layer.")
    # --- 结束 MODIFICATION ---

    args = parser.parse_args()

    eval_model(args)
