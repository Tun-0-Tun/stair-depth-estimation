# Бейзлайны

| Имя | Статья | Код | Веса | Вход | Выход | Запускается |
|---|---|---|---|---|---|---|
| `bicubic` | — | — | — | плотная глубина низкого разрешения | плотная глубина, метры | да |
| `nn_fill` | — | — | — | разреженная глубина | плотная глубина, метры | да |
| `local_bilateral` | — | — | только бэкбон DAv2, тот же файл, что у DuCos | RGB + разреженная глубина, метры | плотная глубина, метры | да |
| DEPTHOR | Xiang et al., DEPTHOR, ICCV 2025, [arXiv:2504.01596](https://arxiv.org/abs/2504.01596) | [ShadowBbBb/Depthor](https://github.com/ShadowBbBb/Depthor) | Google Drive, `gdown` | RGB + разреженный dToF, метры, 0 = нет измерения | плотная глубина, метры | да; CUDA, на остальном через `bpops_shim` |
| DuCos | Yan et al., DuCos, ICCV 2025, [arXiv:2503.04171](https://arxiv.org/abs/2503.04171) | [yanzq95/DuCos](https://github.com/yanzq95/DuCos) | [HF RaynWu2002/DuCos](https://huggingface.co/RaynWu2002/DuCos) плюс бэкбон DAv2 | RGB + плотная глубина низкого разрешения | плотная глубина, метры | да |
| WAVE | Nasir et al., WAVE, 2026, [arXiv:2608.25302](https://arxiv.org/abs/2608.25302) | [tayyabnasir22/WAVE](https://github.com/tayyabnasir22/WAVE) | Google Drive, `gdown`, по файлу на масштаб | RGB + плотная глубина низкого разрешения, только x8/x16/x32 | плотная глубина, метры | да, CUDA |
| DORNet | Wang et al., DORNet, CVPR 2025 oral, [arXiv:2410.11666](https://arxiv.org/abs/2410.11666) | [yanzq95/DORNet](https://github.com/yanzq95/DORNet) | в самом репозитории, [`checkpoints/`](https://github.com/yanzq95/DORNet/tree/main/checkpoints) — приедут вместе с submodule | RGB + глубина низкого разрешения, деградация модели неизвестна | плотная глубина, метры | пока не добавлен|
