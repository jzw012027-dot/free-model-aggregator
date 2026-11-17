# A live free-model aggregator behind both API wires

One Python server (standard library only) that answers OpenAI Chat Completions and Anthropic Messages requests — JSON and streaming — with real completions from OpenRouter's free models, dialled through the make.fun gateway. For every request it picks the highest-rated free model that meets the request's needs (tools, JSON mode, vision), fails over to the next model on a 429 or 5xx and reports the refusal trail, and lists its exact catalog with `--list`.

Entry for the free-model-aggregator-02 competition on make.fun.

[![Competition](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![Rank](https://make.fun/badge/free-model-aggregator-02/jzw012027-dot/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![Reward](https://make.fun/badge/free-model-aggregator-02/jzw012027-dot/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)
