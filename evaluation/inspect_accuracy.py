import os
from tqdm import tqdm
import json
import numpy as np


def extract_pipeline_accuracy(output_folder_path, answer_folder, pipeline_name):
    # baseline
    if not os.path.isdir(output_folder_path):
        print(f"{pipeline_name} accuracy: N/A — cache dir {output_folder_path} does not exist  (pipeline not run yet)")
        return
    video_id_list = sorted(os.listdir(output_folder_path))
    accuracy_list = []
    target_video_id_set = set()
    for video_id in video_id_list:
        subset_accuracy_list = []
        answer_folder_path = os.path.join(output_folder_path, video_id, answer_folder)
        if not os.path.exists(answer_folder_path): continue
        answer_file_list = os.listdir(answer_folder_path)
        target_video_id_set.add(video_id)
        for answer_file in answer_file_list:
            answer_file_path = os.path.join(answer_folder_path, answer_file)
            with open(answer_file_path, 'r') as f:
                answer_dict = json.load(f)
                if answer_dict['is_correct'] is None:
                    accuracy_list.append(0)
                    subset_accuracy_list.append(0)
                else:
                    accuracy_list.append(int(answer_dict['is_correct']))
                    subset_accuracy_list.append(int(answer_dict['is_correct']))
        # print(f"{video_id}: {np.mean(subset_accuracy_list)}")

    if not accuracy_list:
        print(f"{pipeline_name} accuracy: N/A — no cached answers under {output_folder_path}/<video>/{answer_folder}/  (has the pipeline been run?)")
        return
    print(f"{pipeline_name} accuracy: {np.mean(accuracy_list)}  (n_answers={len(accuracy_list)})")


def main():
    extract_pipeline_accuracy(
        "cache/pipeline_baseline",
        "answers_filter_v3_gemini",
        "Baseline",
    )

    extract_pipeline_accuracy(
        "cache/pipeline_a1",
        "answers_filter_q3t_v3_q3vl",
        "A1",
    )

    extract_pipeline_accuracy(
        "cache/pipeline_b1_d1",
        "answers_filter_q3t_v3_gfl",
        "B1",
    )

    extract_pipeline_accuracy(
        "cache/pipeline_c1",
        "answers_filter_q3t_v3_gfl",
        "C1",
    )

    extract_pipeline_accuracy(
        "cache/pipeline_b1_d1",
        "answers_filter_q3t_v3_q3vl",
        "D1",
    )

    extract_pipeline_accuracy(
        "cache/pipeline_b1_d1",
        "answers_filter_q3t_v3_q35",
        "D1-Cans",
    )


if __name__ == '__main__':
    main()