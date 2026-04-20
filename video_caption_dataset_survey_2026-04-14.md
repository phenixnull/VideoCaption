# Video Caption Dataset Survey

Date: 2026-04-14

## Scope and reading notes

- Goal: give you a source-grounded dataset sheet for video caption work, with the classic captioning benchmarks first and enough breadth to reach 20 datasets.
- Priority: official project pages, official papers, and official repos when available.
- Important caveat: several legacy video datasets have multiple counts in the literature because of split conventions, dead YouTube links, cleaned annotations, or challenge-only subsets. I note the most common captioning setup and call out mismatches when they matter.
- `Avg words / vocab` is filled only when the source explicitly reports it or an official benchmark comparison page reports it. Otherwise I mark `N/R`.
- `Metric fit` tells you how naturally the dataset matches your current stack of `CIDEr / BLEU-4 / METEOR / ROUGE-L`.
- Size mix: this sheet keeps the majority in small-to-medium practical benchmarks, then adds heavier or newer sets at the end.

Metric shorthand:

- `B4` = BLEU-4
- `M` = METEOR
- `R` = ROUGE-L
- `C` = CIDEr
- `BS` = BERTScore
- `SODA` = dense-captioning metric used by some newer works
- `VS` = VDCScore

## Quick recommendation for your current roadmap

If you are already running `MSVD` and `MSR-VTT`, the most natural next datasets are:

1. `YouCook2`: small enough to iterate on, but already dense / step-level.
2. `ActivityNet Captions`: the standard jump if you want true dense captioning.
3. `Charades Captions`: detailed short-video captions with longer sentences than MSR-VTT/MSVD.
4. `TVC (TV show Caption)`: good if you want more narrative and multimodal context.

If you want to stay compute-safe for now, do **not** jump directly to `VDC`, `ViCaS`, `S-MiT`, or large weakly supervised corpora like `HowTo100M`.

## Table A. Classic short-clip and sentence-level benchmarks

| Dataset | Year | Size | Domain / task | Official scale | Avg words / vocab | Annotation style | Metric fit | Official source | Download / access | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|
| MSVD / YouTube2Text | 2011 | S | Open-domain short clips; sentence captioning | Original project: `2,089` clips, `85,550` English descriptions, `122k` multilingual descriptions; reconstructed tarball includes `1,970` clips | Clip length usually `<10s`; later official benchmark comparisons report about `8.67` words / caption and `13,010` vocab | Crowd-sourced multilingual one-sentence descriptions | High | [project](https://www.cs.utexas.edu/~ml/clamp/videoDescription/) | [English descriptions](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/AllVideoDescriptions.txt), [video tarball](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar) | Classic low-cost baseline; standard split is usually `1200 / 100 / 670`; some videos are gone because of YouTube rot. |
| MSR-VTT | 2016 | M | Open-domain web videos; sentence captioning | `7,180` source videos, `10,000` clips, `200,000` captions, `41.2` hours, `20` categories | `1,856,523` words, `29,316` vocab, about `9.28` words / caption | AMT; `20` sentences per clip; audio retained | High | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_.pdf) | Official challenge/download portals are legacy-only in many setups; paper remains the stable source | The most common open-domain benchmark after MSVD; a very natural next stop for almost every captioning paper. |
| TGIF | 2016 | M | Animated GIFs / short visual event description | Official paper reports `100k` animated GIFs and `120k` natural-language descriptions | N/R in abstract | Crowdsourced single-sentence GIF descriptions | High | [paper](https://arxiv.org/abs/1604.02748) | See dataset release linked from the paper / legacy project pages | Good if you want very short clips and lighter experimentation, but it is GIF-centric rather than full video. |
| VATEX | 2019 | M | Open-domain multilingual captioning | `41,250` videos, `825,000` captions, `206,000+` EN-ZH parallel pairs | Official benchmark comparison pages report about `12.09` words / caption and `44,103` vocab | Human English and Chinese captions; multilingual and translation-friendly | High | [paper](https://arxiv.org/abs/1904.03493) | See official VATEX site/repo linked from the paper | Strong choice if you want a bigger open-domain benchmark while staying inside the standard `B4/M/R/C` ecosystem. |
| Charades Captions | 2018 | M | Fine-grained daily indoor activities; detailed captioning | Built on `9,848` Charades videos; paper introduces a large detailed-caption variant and reports average caption length `24.13` words | `24.13` words / caption reported in the introducing paper | Detailed multi-sentence / longer-form crowd captions over human activities | High | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Video_Captioning_via_CVPR_2018_paper.pdf) | Built on top of the Charades base dataset; see the paper plus Charades official site/release | One of the best fits if you specifically want captions around the `20-30` word regime. |
| BDD-X | 2018 | M | Driving videos with descriptions and explanations | `6,970` videos, `77+` hours, `26k+` action segments, average video length about `40s` | N/R in official repo stats | Timestamped driving descriptions plus textual explanations of why the driver acts | Medium | [official repo](https://github.com/JinkyuKimUCB/BDD-X-dataset) | [repo download instructions](https://github.com/JinkyuKimUCB/BDD-X-dataset) | Useful if you want captioning plus rationale generation in autonomous-driving scenes. |
| ST-Caps | 2025 | M | Scene-text-aware video captioning | `20,715` videos and `82k+` descriptions | N/R in snippet I could verify | Human captions guided by scene text appearing in video | Medium | [paper landing page](https://www.sciencedirect.com/science/article/abs/pii/S0957417425014538) | Access path depends on publisher / authors; paper is the stable source | Interesting if OCR or scene text matters for your downstream setup. |

## Table B. Dense, step-level, or long-form practical benchmarks

| Dataset | Year | Size | Domain / task | Official scale | Avg words / vocab | Annotation style | Metric fit | Official source | Download / access | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|
| ActivityNet Captions | 2017 | M | Open-domain dense captioning | `20k` videos, `849` video hours, `100k` temporally localized descriptions | Supplementary material reports `13.48 +/- 6.33` words / sentence, `3.65` sentences / video; official benchmark comparison pages report vocab about `15,564` | Workers first write a paragraph, then timestamp each sentence | Very high | [paper](https://arxiv.org/abs/1705.00754), [supplement](https://openaccess.thecvf.com/content_ICCV_2017/supplemental/Krishna_Dense-Captioning_Events_in_ICCV_2017_supplemental.pdf) | [ActivityNet download](https://activity-net.org/download.html) | This is the standard dense-captioning jump from MSR-VTT/MSVD. |
| YouCook2 | 2018 | S | Instructional cooking; step-level captioning | `2,000` long untrimmed videos from `89` recipes, `176` hours total; official split `1333 / 457 / 210` videos | Official site says vocabulary `2600+`; official comparison pages report about `7.88` words / caption and vocab about `2,583` | Human imperative step descriptions with start/end timestamps | Very high | [official site](https://youcook2.eecs.umich.edu/), [paper](https://arxiv.org/abs/1703.09788) | [download page](https://youcook2.eecs.umich.edu/download) | Probably the safest next dataset after MSR-VTT/MSVD if you want dense captions without exploding compute. |
| TV show Caption (TVC) | 2020 | M | TV episodes; multimodal captioning over annotated moments | Built on TVR: `109k` annotated moments / queries on `21.8k` videos from `6` TV shows; TVC adds `262k` captions | N/R in abstract | Additional moment-level descriptions collected over TVR windows; subtitle-aware setup | High | [paper / TVR abstract with TVC release](https://arxiv.org/abs/2001.09099) | [dataset page](https://tvr.cs.unc.edu/tvc.html) | Strong if you want captioning with dialogue/subtitle context and richer narrative structure. |
| ViTT (Video Timeline Tags) | 2020 | M | Instructional videos; dense captioning with short free-text tags | `8,169` videos and `12,461` annotation sets | Tags are intentionally short free-text timeline labels; no single official average word count reported in the README | Segment-level timestamp + concise free-text tag | Medium to High | [paper](https://arxiv.org/abs/2011.11760), [official repo](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT) | [annotations + split files](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT) | Good if you want dense captioning but with shorter, timeline-style labels than ActivityNet or YouCook2. |
| SoccerNet-Caption | 2023 | M | Soccer broadcast dense commentary generation | `36,894` timestamped commentaries across `715.9` hours of broadcasts | Official paper reports average caption length `21.38` words | Single-timestamp soccer commentaries; broadcast-specific language | Medium | [paper](https://arxiv.org/abs/2304.04565), [challenge repo](https://github.com/SoccerNet/sn-caption) | [SoccerNet downloader / task page](https://www.soccer-net.org/data) | One of the cleanest public datasets near your target `20-30` word length, but the language is highly domain-specific. |
| Traffic Video Captioning (traffic-TVC) | 2024 | S | ADAS / traffic scene captioning | `2,000` traffic videos | N/R in repo snippet | Driving-scene captioning with manually curated traffic scenarios | Medium | [official repo](https://github.com/liuchunsense/TVC-dataset) | [request-based access](https://github.com/liuchunsense/TVC-dataset) | Useful if you want a small, domain-focused traffic caption benchmark. |
| ViCaS | 2025 | M | Detailed captioning + grounded segmentation | Official repo reports annotations for `20,416` videos in v1.0 | N/R in repo summary | Detailed captions plus phrase-grounded, pixel-accurate segmentation masks | Low to Medium | [official repo](https://github.com/Ali2500/ViCaS) | [repo / linked Hugging Face resources](https://github.com/Ali2500/ViCaS) | Strong new benchmark, but heavier than the classic sentence-level sets and not the first thing I would train if compute is tight. |
| VDC | 2024 | S | Detailed long-form video captioning benchmark | `1,027` videos / clips / captions | `500.91` words / caption, `20,419` vocab | Structured captions: short, background, main object, camera, detailed caption | Low | [project page](https://rese1f.github.io/aurora-web/) | [benchmark](https://huggingface.co/datasets/wchai/AuroraCap-VDC), [data](https://huggingface.co/datasets/wchai/AuroraCap-Trainset) | Excellent for very long detailed captioning, but clearly much heavier than what you said you want to run now. |

## Table C. Movie, TV, and narration-style datasets

| Dataset | Year | Size | Domain / task | Official scale | Avg words / vocab | Annotation style | Metric fit | Official source | Download / access | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|
| M-VAD | 2015 | M | Movie description from DVS audio | `92` DVDs, `84.6` hours of paired video/sentence data; comparison tables report `48,986` clips and `55,905` sentences | Official comparison tables report `519,933` words, `18,269` vocab, about `9.30` words / caption | Descriptive Video Service narration aligned to movie clips | Medium | [paper](https://arxiv.org/abs/1503.01070) | Access is usually through legacy challenge / author-distributed links | Good movie-domain benchmark, but much noisier and more domain-specific than MSR-VTT or YouCook2. |
| MPII-MD | 2015 | M | Movie description from DVS + scripts | Original paper abstract: `54,000+` sentences and snippets from `72` HD movies; benchmark comparison tables report `94` movies, `68,337` clips, `68,375` sentences | Official comparison tables report `653,467` words, `24,549` vocab, about `9.56` words / caption | Aligned DVS plus movie scripts | Medium | [paper](https://arxiv.org/abs/1501.02530) | Usually obtained through the movie-description / LSMDC access flow | Very common historical movie benchmark; LSMDC is the larger downstream consolidation. |
| LSMDC | 2015-2021 | L | Large-scale movie description challenge | Official challenge page reports `200` movies with `120k` corresponding sentence descriptions | Challenge pages note many clips are short `4-5s`; long-caption paraphrases are also provided for subsets | Challenge dataset built on MPII-MD and M-VAD; movie-description, retrieval, and related tasks | Medium | [download page](https://sites.google.com/site/describingmovies/download), [challenge overview](https://sites.google.com/site/iccv19clvllsmdc/lsmdc-challenge) | [LSMDC download](https://sites.google.com/site/describingmovies/download) | Historically important, but raw access is more cumbersome than MSR-VTT / YouCook2 / ActivityNet. |
| TV show Caption (TVC) | 2020 | M | TV-show moments with captions | `262k` captions on top of TVR moment annotations | N/R | Moment-level captions over TV episodes, often used with subtitle context | High | [paper](https://arxiv.org/abs/2001.09099) | [dataset page](https://tvr.cs.unc.edu/tvc.html) | Repeated here because it sits halfway between dense captioning and TV/movie narration. |
| Spoken Moments in Time (S-MiT) | 2021 | XL | Spoken video description corpus | Official comparison pages report `515,912` video-caption pairs | Official comparison pages report `5,618,064` words, `50,570` vocab, about `10.89` words / caption | Spoken descriptions aligned with open-domain video clips | Low to Medium | [paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Monfort_Spoken_Moments_Learning_Joint_Audio-Visual_Representations_From_Video_Descriptions_CVPR_2021_paper.pdf) | See project / release linked from the paper | Attractive for scale, but this is more of a large spoken-caption corpus than a lightweight benchmark. |

## Table D. Large-scale narrated or weakly supervised corpora

| Dataset | Year | Size | Domain / task | Official scale | Avg words / vocab | Annotation style | Metric fit | Official source | Download / access | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|
| HowTo100M | 2019 | XL | Instructional web video pretraining corpus | `1.22M` narrated instructional videos and `136M` clips; `23k+` visual tasks | Comparison pages report about `4.16` words / clip and very large noisy vocabulary | Automatic clip-text pairs from ASR narrations; weak supervision | Low | [paper](https://arxiv.org/abs/1906.03327) | [project page](http://www.di.ens.fr/willow/research/howto100m/) | Great for transfer/pretraining; poor first choice if you want clean CIDEr-style caption benchmarking. |
| Spoken Moments in Time (S-MiT) | 2021 | XL | Open-domain spoken captions | `515,912` video-caption pairs | About `10.89` words / caption and `50,570` vocab on official benchmark comparison pages | Spoken descriptions, not the same annotation flavor as classic sentence-caption datasets | Low to Medium | [paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Monfort_Spoken_Moments_Learning_Joint_Audio-Visual_Representations_From_Video_Descriptions_CVPR_2021_paper.pdf) | See paper resources | Included again here because in practice it behaves more like a large corpus than a small benchmark. |

## Selection guidance by compute and similarity to your current setup

### Best next datasets for you

| Priority | Dataset | Why it matches your current pipeline |
|---|---|---|
| 1 | YouCook2 | Small enough to train quickly, still dense/step-level, standard `B4/M/R/C` reporting, and much closer to a manageable `CLIP4Clip`-era next step than VDC. |
| 2 | ActivityNet Captions | The canonical dense-captioning benchmark; if you later say "we support dense captioning" this is the name reviewers expect to see. |
| 3 | Charades Captions | Longer and richer captions than MSVD/MSR-VTT without the huge overhead of detailed-caption benchmarks. |
| 4 | TVC | Stronger narrative and multimodal context than the very short classic benchmarks. |
| 5 | SoccerNet-Caption | Only if you are okay with highly domain-specific language; average sentence length is very close to your target. |

### Good but more specialized

| Dataset | Why it is useful | Why I would not make it your first expansion |
|---|---|---|
| BDD-X | Description + explanation is useful for explainable driving or rationale generation. | Narrow driving domain; not as standard for generic video caption claims. |
| ViTT | Clean dense timeline tags on instructional videos. | Tags are often shorter and more telegraphic than full captions. |
| Traffic TVC | Small domain benchmark for traffic captioning. | Niche; less comparable to mainstream caption papers. |
| ST-Caps | Good if you care about scene text. | Not yet part of the mainstream classic benchmark bundle. |

### Datasets I would avoid for now if your goal is stable progress

| Dataset | Why to avoid right now |
|---|---|
| VDC | Captions are extremely long; this is a different difficulty class from MSR-VTT / MSVD. |
| ViCaS | Strong benchmark, but the segmentation grounding makes the ecosystem heavier than you need for a first extension. |
| S-MiT | Much larger and more corpus-like than the classic benchmark flow you are currently on. |
| HowTo100M | Best treated as a pretraining source, not as a clean evaluation benchmark. |

## Metric compatibility notes

- If you want the most apples-to-apples continuation of `MSVD / MSR-VTT` with `CIDEr + BLEU-4 + METEOR + ROUGE-L`, the safest set is:
  - `YouCook2`
  - `ActivityNet Captions`
  - `Charades Captions`
  - `VATEX`
  - `TVC`
  - `ViTT`
- Movie-domain sets (`M-VAD`, `MPII-MD`, `LSMDC`) are historically important, but many older papers emphasize `METEOR` more than modern open-domain caption papers do.
- Newer detailed-caption sets (`VDC`, some long-caption benchmarks) increasingly add:
  - `BERTScore`
  - `CLIPScore`
  - `LLM-as-judge`
  - benchmark-specific metrics such as `VDCScore`

## Practical takeaway

If I were choosing the next 4 datasets for your exact situation, I would use:

1. `YouCook2`
2. `ActivityNet Captions`
3. `Charades Captions`
4. `TVC`

That combination keeps you close to the classic captioning literature, stays compatible with your current metrics, adds both dense and longer-caption settings, and avoids the "too long / too heavy" trap of the newest detailed-caption benchmarks.
