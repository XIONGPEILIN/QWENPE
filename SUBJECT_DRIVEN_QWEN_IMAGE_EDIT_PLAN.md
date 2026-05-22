# Qwen Image Edit Subject-Driven 数据生成计划

## 1. 当前任务定义

当前任务只保留一个输入条件：

```text
一张已经抠出来的 subject reference 图片
```

目标是用 `Qwen Image Edit` 生成：

```text
同一个 subject
+ 不同背景 / 不同环境 / 不同构图
= 新的完整图片
```

当前主方案已经固定为：

```text
输入只有单张 reference subject 图；
背景不作为第二张输入图提供；
背景通过 prompt 描述；
Qwen Image Edit 直接生成完整新图。
```

## 2. 已明确放弃的路线

下面这条路线已经不再使用：

```text
背景图 + reference 图双输入
```

放弃原因只有一个：

```text
当前实测效果差，生成质量不稳定。
```

因此后续所有计划、脚本、prompt 和数据格式，全部以“单 reference 输入”方案为准。

## 3. 任务的核心约束

生成时必须同时满足以下要求：

1. subject 必须看起来是同一个实例，不是同类物体。
2. 新图的背景必须明显不同于原始背景。
3. subject 必须尽量占满画面，至少是主要视觉主体。
4. 模型应主要改变背景、环境、光照和构图，而不是改写 subject 本身。
5. 结果应适合作为后续训练或评测数据。

## 4. 文献依据与执行结论

当前 prompt 方案按下面四条文献结论固定。

1. DreamBooth 的核心做法是固定 subject 标识，然后把它放进不同 context sentence 里生成不同环境。
2. BLIP-Diffusion 的核心做法是让 reference image 负责 subject fidelity，让短文本 prompt 负责 context，并在训练里用“同一 subject 不同 context”与随机背景组合。
3. FastComposer 的核心做法是保持 subject conditioning 稳定，同时用 text instruction 去改变 style、action 和 context。
4. Qwen Image Edit 官方模型卡已经把背景修改列为支持场景，因此当前可执行路线就是“固定 subject 约束 + 模板化背景 prompt”。

这四条一起导出的执行结论是：

```text
固定 subject 相关句子；
只系统性地改变背景 context；
不要每次自由改写整条 prompt。
```

## 5. 默认固定 prompt

当前唯一有效的默认模板是：

```text
Picture 1 is the image to modify. Keep the exact same subject from the input image unchanged. Preserve the same object identity, silhouette, colors, texture, markings, and proportions. Keep the subject as the main focus and make it occupy most of the frame. Place the subject in a photorealistic scene with [scene_background]. Only change the environment, lighting, and composition so the subject appears naturally placed in [target_scene].
```

这个模板现在固定不变。

当前只允许替换两个槽位：

1. `[scene_background]`
2. `[target_scene]`

## 6. 两个槽位的填写规则

### 6.1 `[scene_background]`

`[scene_background]` 必须写成完整、具体、可落地的背景短语。

它必须包含：

1. 场景本体
2. 支撑表面或环境空间
3. 光照或时间信息

它不能包含：

1. subject 名称
2. 主体动作描述
3. 模糊审美词

合格示例：

```text
a clean wooden table in a sunlit room
a rainy neon-lit alley at night
a seamless white studio backdrop with soft lighting
```

不合格示例：

```text
a beautiful background
an amazing scene
a stylish place
```

### 6.2 `[target_scene]`

`[target_scene]` 必须是前一个槽位的简短归纳版本。

它的作用不是引入新场景，而是收束前一句。

合格示例：

```text
the sunlit tabletop scene
the rainy neon-lit alley
the white studio setting
```

规则：

1. `[target_scene]` 必须和 `[scene_background]` 语义一致。
2. `[target_scene]` 不允许引入另一个新地点。
3. `[target_scene]` 必须比 `[scene_background]` 更短。

## 7. 背景多样性的设计方式

背景多样性不靠重写 subject 句子获得，而靠系统变化 context 获得。

当前只允许沿四个维度做多样化：

1. 场景类别
2. 光照时间
3. 空间材质
4. 镜头构图

但第一版执行时，变化规则要收紧：

1. subject 句子完全固定。
2. `photorealistic` 固定。
3. 每次只变背景 context，不变 subject 约束。
4. 第一版不引入夸张风格词。
5. 第一版不同时叠加过多天气、动作、叙事元素。

## 8. 第一版固定背景 prompt bank

第一版直接固定 12 条背景 prompt bank，不现场生成。

### 8.1 tabletop

```json
{"background_prompt_id":"tabletop_001","scene_background":"a clean wooden table in a sunlit room","target_scene":"the sunlit tabletop scene"}
{"background_prompt_id":"tabletop_002","scene_background":"a minimalist white desk near a large window with soft daylight","target_scene":"the minimalist desk scene"}
```

### 8.2 indoor

```json
{"background_prompt_id":"indoor_001","scene_background":"a modern living room with soft morning light","target_scene":"the living room scene"}
{"background_prompt_id":"indoor_002","scene_background":"a clean kitchen interior with natural daylight","target_scene":"the kitchen interior scene"}
```

### 8.3 outdoor_nature

```json
{"background_prompt_id":"nature_001","scene_background":"a sandy beach at sunset","target_scene":"the beach scene at sunset"}
{"background_prompt_id":"nature_002","scene_background":"a grassy outdoor field under soft daylight","target_scene":"the grassy outdoor scene"}
```

### 8.4 street

```json
{"background_prompt_id":"street_001","scene_background":"a rainy neon-lit alley at night","target_scene":"the rainy neon-lit alley"}
{"background_prompt_id":"street_002","scene_background":"a quiet city sidewalk in soft afternoon light","target_scene":"the city sidewalk scene"}
```

### 8.5 display

```json
{"background_prompt_id":"display_001","scene_background":"a clean exhibition pedestal under studio lighting","target_scene":"the exhibition display scene"}
{"background_prompt_id":"display_002","scene_background":"a product display platform with soft controlled shadows","target_scene":"the product display scene"}
```

### 8.6 studio

```json
{"background_prompt_id":"studio_001","scene_background":"a seamless white studio background with soft lighting","target_scene":"the white studio setting"}
{"background_prompt_id":"studio_002","scene_background":"a neutral gray studio backdrop with soft shadows","target_scene":"the gray studio setting"}
```

## 9. prompt 生成规则

每条最终 prompt 必须满足以下规则：

1. 固定句子完全不改。
2. 只从固定 prompt bank 中取槽位。
3. 不现场用 LLM 改写 prompt。
4. 不使用空泛背景词。
5. 不重新用长句描述 subject。
6. 第一版每个 subject 只配 6 条背景 prompt。
7. 这 6 条应当覆盖 6 个不同类别。

## 10. 数据格式

当前主数据格式按单输入方案固定为：

```json
{
  "item_id": "sd_000000_00",
  "prompt": "Picture 1 is the image to modify. Keep the exact same subject from the input image unchanged. Preserve the same object identity, silhouette, colors, texture, markings, and proportions. Keep the subject as the main focus and make it occupy most of the frame. Place the subject in a photorealistic scene with a clean wooden table in a sunlit room. Only change the environment, lighting, and composition so the subject appears naturally placed in the sunlit tabletop scene.",
  "image": "generated/sd_000000_00.png",
  "edit_image": [
    "references/ref_000000.png"
  ],
  "ref_gt": "references/ref_000000.png",
  "global_caption": "...",
  "local_caption": "...",
  "source_index": 0,
  "variant_id": 0,
  "seed": 420000,
  "background_prompt_id": "scene_01357",
  "background_prompt_text": "...",
  "generation_status": "pending"
}
```

字段解释：

1. `image` 是最终生成结果。
2. `edit_image[0]` 是唯一输入，也就是 reference subject 图。
3. `ref_gt` 继续指向 reference 图，方便保留统一语义。
4. `background_prompt_text` 记录实际使用的背景描述。
5. `seed` 记录采样随机种子。

## 11. 目录结构

当前目录结构简化为：

```text
subject_driven_qwen_edit/
  openimages/
    references/
      ref_000000.png
      ref_000000_mask.png
      ref_000000_rgba.png
    generated/
      sd_000000_00.png
    generated_sub/
      sd_000000_00.png
    debug/
      sd_000000_00/
        reference.png
        output.png
        output_crop.png
        prompt.txt
        metadata.json
  metadata/
    subject_records.json
    prompt_bank.json
    subject_driven_manifest.json
    subject_driven_generation_log.jsonl
    subject_driven_quality_scores.jsonl
  dataset_qwen_subject_driven_all.json
  dataset_qwen_subject_driven_train.json
  dataset_qwen_subject_driven_val.json
  dataset_qwen_subject_driven_test.json
```

## 12. 执行流程

当前有效流程只有六步：

1. 检查 reference 数据是否完整可读。
2. 标准化 reference 图。
3. 为每个 reference 生成稳定 subject caption。
4. 从背景模板库生成 prompt bank。
5. 用单 reference 输入 + 固定 prompt 模板批量生成图片。
6. 做自动筛选和人工抽查。

## 13. prompt bank 的构造方式

`prompt_bank.json` 应该保存可重复使用的背景模板。

每条记录示例：

```json
{
  "background_prompt_id": "scene_01357",
  "scene_background": "a clean wooden table in a sunlit room",
  "target_scene": "the sunlit tabletop scene",
  "category": "tabletop",
  "full_prompt": "Picture 1 is the image to modify. Keep the exact same subject from the input image unchanged. Preserve the same object identity, silhouette, colors, texture, markings, and proportions. Keep the subject as the main focus and make it occupy most of the frame. Place the subject in a photorealistic scene with a clean wooden table in a sunlit room. Only change the environment, lighting, and composition so the subject appears naturally placed in the sunlit tabletop scene."
}
```

生成最终 prompt 时有两种允许方式：

1. 直接读取 `full_prompt`。
2. 用固定模板对 `scene_background` 和 `target_scene` 做字符串填槽。

不允许第三种方式：

```text
运行时再调用模型重写 prompt
```

## 14. 生成脚本要求

新脚本应按当前路线实现：

```text
qwen_cli_subject_driven_single_ref.py
```

脚本要求：

1. 每条任务只加载一张 `reference` 图。
2. 读取 manifest 中固定好的完整 prompt。
3. 将 `edit_image=[reference_pil]` 传给 Qwen。
4. 不再传背景图。
5. 不再依赖插入 mask。
6. 输出最终图和调试信息。

核心调用应接近：

```python
output_image = pipe(
    prompt=prompt,
    edit_image=[reference_pil],
    height=target_h,
    width=target_w,
    num_inference_steps=args.steps,
    cfg_scale=args.cfg_scale,
    seed=seed,
)[0]
```

如果当前 Qwen 版本返回多张结果，则按实际返回接口处理，但单输入原则不变。

## 15. 多 seed 策略

多 seed 的作用只有两个：

1. 增加背景细节多样性。
2. 增加构图和融合细节多样性。

多 seed 不是用来替代 subject identity 约束的。

第一版建议：

1. 每个 subject x prompt 组合采样 2 到 4 个 seed。
2. 保留 identity 最稳定的一张。
3. 如果四个 seed 都失败，则标记该 prompt 不稳定。

## 16. 自动筛选标准

自动筛选至少要检查三类问题：

### 16.1 subject identity

检查生成结果是否仍然像 reference 里的同一个 subject。

### 16.2 subject 占画面比例

检查生成结果里 subject 是否仍然是主视觉主体。

### 16.3 背景可辨识度

检查背景是否真的发生了明显变化，而不是只是原图轻微改写。

## 17. 第一版实验建议

第一版不要全量跑，先做小规模验证。

建议规模：

1. 先选 100 个 subject。
2. 每个 subject 配 6 条背景 prompt。
3. 每条 prompt 跑 2 个 seed。
4. 总共先生成 1200 张。

这样可以先验证三件事：

1. identity 是否稳定。
2. subject 是否真的占满画面。
3. 哪些背景模板最容易失败。

## 18. 当前最重要的工程原则

1. 只保留一个输入图。
2. prompt 固定结构，不做自由改写。
3. subject identity 相关句子固定。
4. subject 占满画面这句固定。
5. 背景描述通过模板库统一生成。
6. 先做小规模验证，再扩大量级。

## 19. 当前默认执行结论

如果现在立刻开始跑数据，默认执行结论就是：

```text
输入一张 reference subject 图；
使用固定 prompt 模板；
只替换背景槽位；
让 Qwen Image Edit 直接生成完整新图；
再通过多 seed 和筛选保留最稳定结果。
```

## 20. 参考依据

1. DreamBooth project page: https://dreambooth.github.io/
2. DreamBooth paper: https://arxiv.org/abs/2208.12242
3. BLIP-Diffusion project page: https://dxli94.github.io/BLIP-Diffusion-website/
4. BLIP-Diffusion paper: https://arxiv.org/abs/2305.14720
5. FastComposer paper: https://arxiv.org/abs/2305.10431
6. Qwen Image Edit model card: https://huggingface.co/Qwen/Qwen-Image-Edit
7. Qwen Cloud image model docs: https://docs.qwencloud.com/developer-guides/getting-started/image-models
