from src.run_model import load_model, run_model
from src.load_data import load_benchmark
from src.eval_module import EvaluationMaster, EvalModule, SimpleAnswerProcessor, AccuracyEvaluator
from src.cache_sys import AnswerCacheSystem, get_cache_id
from argparse import ArgumentParser
from tqdm import tqdm

def main(args):
    benchmark_data = load_benchmark(args.video_dir, args.mode, args.questions_dir)
    model_manager = load_model(args.model_id, args.debug_with_n_frames)
    cache_system = AnswerCacheSystem(args.model_id, args.cache_dir)
    eval_master = EvaluationMaster([
        EvalModule("accuracy", processor=SimpleAnswerProcessor(), evaluator=AccuracyEvaluator()),
    ])
    for idx, benchmark_sample in enumerate(tqdm(benchmark_data)):
        cache_id = get_cache_id(benchmark_sample['video_name'])
        if not cache_system.exist(cache_id):
            answers = run_model(model_manager, args.model_id, benchmark_sample, sample_id=idx, debug_with_n_frames=args.debug_with_n_frames)
            cache_system.push(cache_id, answers)
        else:
            answers = cache_system.get(cache_id)
        eval_master.batch_push(predictions = answers, truths = benchmark_sample['answers'])

    result = eval_master.compute_result()
    print(result)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run model against benchmark")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--video_dir", type=str, default="raw_data")
    parser.add_argument("--questions_dir", type=str, default="benchmark")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--debug_with_n_frames", type=int, default=None)
    args = parser.parse_args()
    main(args)