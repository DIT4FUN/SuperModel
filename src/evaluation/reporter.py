"""
评估报告生成器
==============

生成标准化的评估报告 (JSON/Markdown/HTML)
支持 AGV 五级合规性判定
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .benchmark import BenchmarkResult, BenchmarkSuite
from .metrics import LatencyMetrics, MultimodalMetrics, ControlMetrics


class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, grade: str, output_dir: str = "evaluation_reports"):
        self.grade = grade
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().isoformat()
        self.results: List[BenchmarkResult] = []
    
    def add_results(self, results: List[BenchmarkResult]):
        self.results.extend(results)
    
    def generate_json(self, path: Optional[str] = None) -> str:
        """生成 JSON 格式报告"""
        data = {
            "grade": self.grade,
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "pass_rate": sum(1 for r in self.results if r.passed) / max(len(self.results), 1),
            "results": [r.to_dict() for r in self.results],
        }
        
        path = path or str(self.output_dir / f"eval_{self.grade}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
    
    def generate_markdown(self, path: Optional[str] = None) -> str:
        """生成 Markdown 格式报告"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        pass_rate = 100 * passed / max(total, 1)
        
        lines = [
            f"# SuperModel 评估报告",
            f"",
            f"**AGV 等级**: {self.grade}",
            f"**时间**: {self.timestamp}",
            f"**通过率**: {passed}/{total} ({pass_rate:.1f}%)",
            f"",
            f"## 详细结果",
            f"",
            f"| 模块 | 延迟 (ms) | 规格 (ms) | 内存 (MB) | FPS | 准确率 | 状态 |",
            f"|------|----------|----------|---------|-----|--------|------|",
        ]
        
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"| {r.name} | {r.latency_ms:.2f} | {r.latency_spec_ms:.2f} | "
                        f"{r.memory_mb:.1f} | {r.throughput_fps:.1f} | {r.accuracy:.2%} | {status} |")
        
        lines.append("")
        lines.append("## 合规性判定")
        lines.append("")
        
        for r in self.results:
            if not r.passed:
                lines.append(f"⚠️ **{r.name}**: 延迟超标 ({r.latency_ms:.2f}ms > {r.latency_spec_ms:.2f}ms)")
        
        if all(r.passed for r in self.results):
            lines.append(f"🎉 **{self.grade} 等级全部指标合规!**")
        
        path = path or str(self.output_dir / f"eval_{self.grade}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        content = "\n".join(lines)
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def generate_html(self, path: Optional[str] = None) -> str:
        """生成 HTML 格式报告"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        pass_rate = 100 * passed / max(total, 1)
        
        result_rows = []
        for r in self.results:
            status_icon = "✅" if r.passed else "❌"
            result_rows.append(f"""
            <tr>
                <td>{r.name}</td>
                <td>{r.latency_ms:.2f}</td>
                <td>{r.latency_spec_ms:.2f}</td>
                <td>{r.memory_mb:.1f}</td>
                <td>{r.throughput_fps:.1f}</td>
                <td>{r.accuracy:.2%}</td>
                <td>{status_icon}</td>
            </tr>
            """)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SuperModel 评估报告 - {self.grade}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .pass {{ color: green; }} .fail {{ color: red; }}
    </style>
</head>
<body>
    <h1>🤖 SuperModel 评估报告</h1>
    <div class="summary">
        <p><strong>AGV 等级:</strong> {self.grade}</p>
        <p><strong>时间:</strong> {self.timestamp}</p>
        <p><strong>通过率:</strong> {passed}/{total} ({pass_rate:.1f}%)</p>
    </div>
    
    <h2>详细结果</h2>
    <table>
        <tr>
            <th>模块</th><th>延迟 (ms)</th><th>规格 (ms)</th>
            <th>内存 (MB)</th><th>FPS</th><th>准确率</th><th>状态</th>
        </tr>
        {''.join(result_rows)}
    </table>
</body>
</html>
        """
        
        path = path or str(self.output_dir / f"eval_{self.grade}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(path, 'w') as f:
            f.write(html)
        return path
    
    def generate_all(self) -> Dict[str, str]:
        """生成所有格式报告"""
        return {
            "json": self.generate_json(),
            "markdown": self.generate_markdown(),
            "html": self.generate_html(),
        }
