"""Complete reporting after the scientific queue, surviving chat interruptions."""
from common import *
import argparse
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager-pid', type=int, required=True)
    cli = parser.parse_args()
    progress = {'status': 'waiting_for_main_queue', 'started_at': time.time()}
    status_path = ROOT / 'postprocess-status.json'
    write_json(status_path, progress)
    try:
        while True:
            state = json.loads((ROOT / 'status.json').read_text())
            if state['status'] == 'complete':
                break
            if state['status'] in ['failed', 'needs_diagnosis']:
                raise RuntimeError(f'Main queue needs diagnosis: {state}')
            os.kill(cli.manager_pid, 0)
            time.sleep(5)
        for filename in ['analyze_results.py', 'plot_results.py', 'collect_examples.py',
                         'final_audit.py', 'write_report.py']:
            progress.update(status='postprocessing', active_script=filename)
            write_json(status_path, progress)
            print('POSTPROCESS', filename, flush=True)
            subprocess.run([sys.executable, str(ROOT / filename)], check=True)
        today = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
        plan_path = ROOT / 'study-plan.json'
        plan = json.loads(plan_path.read_text())
        plan.update(status='complete', execution_completed=today,
                    results_file='results.json', report='../../LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md')
        write_json(plan_path, plan)
        docs = WORKSPACE / 'LearningDocs'
        design = docs / '06_4090_EXPERIMENT_DESIGN.md'
        lines = design.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith('日期：'))
        lines[index] = f'日期：2026-10-06。执行完成：{today}。状态：**六个训练 run、600 次开发评估和 600 次最终测试已完成；结果见 [07](07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)。**'
        design.write_text('\n'.join(lines) + '\n')
        navigation = docs / 'README.md'
        lines = navigation.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith('6. [06_'))
        lines[index] = '6. [06_4090_EXPERIMENT_DESIGN.md](06_4090_EXPERIMENT_DESIGN.md)：已执行的 4090 实验方案；10/50 条示范 × 3 个训练 seed、固定计算预算、吞吐标定和独立最终测试。'
        if not any(line.startswith('7. [07_') for line in lines):
            lines.insert(index + 1, '7. [07_PLACEMUGRACK_EXPERIMENT_RESULTS.md](07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)：六组训练的最终成功率、稳定性、失败阶段、资源计时、图件及权重/视频索引。')
        navigation.write_text('\n'.join(lines) + '\n')
        research = WORKSPACE / 'IMITATOR_RESEARCH_PLAN.md'
        text = research.read_text().replace('版本：v1.3', '版本：v1.4')
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith('当前状态：'))
        lines[index] = '当前状态：已完成源码梳理、`imitator` 环境配置、最小训练与评估闭环，以及 PlaceMugRack 本地 ACT 的 10/50 条示范 × 3 个训练 seed 实验。六组均完成 18,000 次更新、共 600 次开发评估及 600 次独立最终测试，权重、指标与录像审计通过。执行方案见 [`LearningDocs/06_4090_EXPERIMENT_DESIGN.md`](LearningDocs/06_4090_EXPERIMENT_DESIGN.md)，结果见 [`LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md`](LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)。本次是单任务小规模 ACT baseline，尚未复现论文整体性能。已发布的 GitHub 仓库不包含本轮新增结果；环境配置和依赖例外见 [`environment/README.md`](environment/README.md)。'
        research.write_text('\n'.join(lines) + '\n')
        progress.update(status='complete', active_script=None, finished_at=time.time(), report=str(docs/'07_PLACEMUGRACK_EXPERIMENT_RESULTS.md'))
        write_json(status_path, progress)
        print('POSTPROCESS COMPLETE', flush=True)
    except Exception as error:
        progress.update(status='failed', error=str(error), finished_at=time.time())
        write_json(status_path, progress)
        raise


if __name__ == '__main__':
    main()
