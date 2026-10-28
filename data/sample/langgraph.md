# LangGraph：把 Agent 写成状态机

## 为什么需要

手写 while 循环的 Agent 难以扩展：加分支、加重试、加人工确认都要改循环体。
LangGraph 把流程抽象成图：状态（State）+ 节点（Node）+ 边（Edge）。

## 核心概念

- StateGraph：用 TypedDict 定义状态，节点返回状态的部分更新；
- 节点：普通函数，做一步工作（调模型、执行工具、写日志）；
- 边：普通边决定下一步；条件边 conditional edge 根据状态决定走哪条路；
- 检查点 checkpointer：每步把状态存下来，支持断点续跑和多轮记忆，
  常用 MemorySaver（内存）或 SqliteSaver / PostgresSaver（持久化）。

## 常见模式

- ReAct 循环：模型节点 → 工具节点 → 回模型，直到没有工具调用；
- 人工在环 human-in-the-loop：写文件、发邮件等高风险动作前中断，等人工批准；
- supervisor 多智能体：一个调度节点把任务分给多个子 Agent（检索、写代码、审稿），再汇总结果。

## 和手写循环比

优点：可观测（每步状态可查）、可恢复（检查点）、可组合（子图）；
代价：多一层抽象，简单任务会显得重。
面试时讲清「什么场景值得上图」比只会 API 更加分。
