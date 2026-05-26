import json
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--annotation-file', type=str, default=None)
    parser.add_argument('--result-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--summary-output-dir', type=str, default=None)

    return parser.parse_args()


def calculate_accuracy(jsonl_file):
    correct_count = 0
    total_count = 0
    total_not_in_options=0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            category = data['question_id'].split('/')[0]
            correct_answer = data['text']
            # 从完整的text中提取最后一个词！！！！！！！！！！！！！！！！！！！！！！！
            correct_answer = correct_answer.split()[-1].strip(".,;:!?")
            # print(correct_answer, category)
            # first_part = correct_answer.split()[0].strip(".,;:!?")  # 去除常见标点
            # correct_answer = [c for c in first_part if c.isalpha()][0] if first_part else ""

            options = {
                "A": "joy",
                "B": "love",
                "C": "surprise",
                "D": "anger",
                "E": "confusion",
                "F": "fear",
                "G": "sadness",
                "A.": "joy",
                "B.": "love",
                "C.": "surprise",
                "D.": "anger",
                "E.": "confusion",
                "F.": "fear",
                "G.": "sadness",
                "joy": "joy",
                "love": "love",
                "surprise": "surprise",
                "anger": "anger",
                "confusion": "confusion",
                "fear": "fear",
                "sad": "sadness",
                "Joy": "joy",
                "Love": "love",
                "Surprise": "surprise",
                "Anger": "anger",
                "Confusion": "confusion",
                "Fear": "fear",
                "Sad": "sadness",
                
            }
            # 先判断是否在正确选项里面！！！！！！！！！！！！！！！！！！！！！！！
            if correct_answer in options:
                if options[correct_answer] == category: 
                # if correct_answer == category:
                    correct_count += 1
            else:
                print(f"Correct answer {correct_answer} not in options")
                total_not_in_options += 1
            total_count += 1
    
    # 计算正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    return total_count, accuracy,total_not_in_options


args = get_args()
jsonl_file = args.result_file
total_count, accuracy,total_not_in_options = calculate_accuracy(jsonl_file)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Total not in options: {total_not_in_options}")


if args.output_dir is not None:
    output_file = os.path.join(args.output_dir, 'result-emoset.txt')
    with open(output_file, 'w') as f:
        f.write('Samples: {}\nAccuracy on emoset: {:.2f}%\n'.format(total_count, accuracy * 100))
        f.write(f"Total not in options: {total_not_in_options}\n")


if args.summary_output_dir is not None: 
    with open(args.summary_output_dir, 'a') as f_sum:
        f_sum.write('\nSamples: {}\nAccuracy on emoset: {:.2f}%\n'.format(total_count, accuracy * 100))
        f_sum.write(f"Total not in options: {total_not_in_options}\n")

