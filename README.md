# tcc-ufrgs-XJTU-SY

Este repositório publica o código desenvolvido durante o Trabalho de Conclusão de Curso
(TCC) em Engenharia Mecânica da Universidade Federal do Rio Grande do Sul (UFRGS), no semestre
2026/2.

**Título:** Aplicação de técnicas de inteligência artificial na manutenção preditiva de
rolamentos: estudo com o conjunto de dados XJTU-SY

| Informação | Identificação |
|---|---|
| Autor | Augusto Arrojo de Deus |
| Curso | Engenharia Mecânica — UFRGS |
| Área de concentração | Processos de Fabricação |
| Orientadora | Profª. Simone Giovelli Ramires |
| Coorientador | Eng. Sandro Terroso |

## Sobre o projeto

O trabalho investiga manutenção preditiva de rolamentos com o
[XJTU-SY Bearing Dataset](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets), conjunto
público com dados run-to-failure de 15 rolamentos obtidos em ensaios acelerados de degradação.
O pipeline experimental abrange:

- auditoria e validação do conjunto de dados;
- extração de atributos de vibração nos canais horizontal e vertical;
- construção de indicador de saúde por PCA;
- estimação não supervisionada do início da degradação;
- regressão da vida útil remanescente (RUL);
- avaliação de contexto temporal com LSTM causal;
- consolidação estatística no nível do rolamento;
- modelagem de sobrevivência com Kaplan–Meier, Cox e Random Survival Forest.

A questão de pesquisa definida na monografia é:

> Como diferentes estratégias baseadas em Inteligência Artificial podem representar a
> degradação de rolamentos a partir de sinais de vibração e de que forma essas informações
> podem apoiar o prognóstico na manutenção preditiva?

O escopo científico está restrito aos 15 rolamentos run-to-failure do XJTU-SY e às três
condições operacionais estudadas. Os resultados não devem ser generalizados diretamente para
ambientes industriais ou outros conjuntos de dados.

## Estado do projeto

As Fases 1 a 8 foram implementadas e executadas sobre os dados reais. A avaliação preserva a
separação completa entre rolamentos de treino, validação e teste, com seleção de detectores e
modelos realizada antes da aplicação ao conjunto de teste.

## Principais resultados

- A primeira componente principal utilizada como indicador de condição explicou entre
  **84,30% e 96,11%** da variância dos atributos selecionados nas cinco partições.
- Os sinais de vibração representaram a evolução da degradação, mas os ensaios apresentaram
  elevada variabilidade de vida experimental, entre **41 e 2.537 minutos**.
- Na estimativa direta da RUL, o baseline baseado na mediana obteve o menor Macro MAE,
  **269,63 minutos**. O melhor modelo supervisionado foi a LSTM com uma aquisição (`k=1`), com
  **280,03 minutos**.
- Nenhum modelo supervisionado apresentou superioridade estatisticamente significativa sobre o
  baseline de mediana após a correção de Holm.
- Na análise de sobrevivência, o Random Survival Forest apresentou a maior discriminação,
  com concordância IPCW/Uno de **0,639**, e os menores IBS nos horizontes administrativos de
  30, 60, 120 e 240 minutos.
- O modelo de Cox baseado em atributos causais apresentou o menor IBS global, **0,245**.
- Após a correção de Holm, nenhuma diferença entre os modelos de sobrevivência e o estimador de
  Kaplan–Meier permaneceu estatisticamente significativa a 5%; as vantagens observadas são,
  portanto, descritivas.

Esses resultados indicam que caracterizar a degradação é mais viável do que generalizar uma
estimativa absoluta de RUL para rolamentos não vistos. A formulação probabilística fornece uma
representação complementar do prognóstico, mas seus ganhos comparativos permanecem exploratórios.

## Instalação

O projeto requer Python 3.11 ou superior e [uv](https://docs.astral.sh/uv/). Para instalar as
dependências:

```bash
uv sync --group dev
```

## Execução resumida

Os comandos abaixo executam as fases na ordem metodológica:

```bash
uv run python -m xjtu_sy_tcc.cli.audit --config configs/data.yaml
uv run xjtu-sy-build-features --config configs/features.yaml
uv run xjtu-sy-run-phase3 --config configs/phase3.yaml
uv run xjtu-sy-run-phase4 --config configs/phase4.yaml
uv run xjtu-sy-run-phase5 --config configs/phase5.yaml
uv run xjtu-sy-run-phase6 --config configs/phase6.yaml
uv run xjtu-sy-run-phase7 --config configs/phase7.yaml
uv run xjtu-sy-run-phase8 --config configs/phase8.yaml
```

O conjunto XJTU-SY deve estar disponível no diretório local `Data/`. Os dados brutos e os
outputs completos são ignorados pelo Git. Figuras e tabelas selecionadas para apresentação estão
organizadas em [`deliverables/`](deliverables/).

## Documentação

- [Guia técnico completo](docs/technical_guide.md): instalação detalhada, estrutura dos dados,
  comandos, outputs, validações, controles de leakage e limitações.
- [Estrutura do conjunto de dados](docs/dataset_structure.md).
- [Metodologia da Fase 1](docs/phase1_methodology.md).
- [Metodologia da Fase 2](docs/phase2_methodology.md).
- [Metodologia da Fase 3](docs/phase3_methodology.md).
- [Metodologia da Fase 4](docs/phase4_methodology.md).
- [Metodologia da Fase 5](docs/phase5_methodology.md).
- [Metodologia da Fase 6](docs/phase6_methodology.md).
- [Metodologia da Fase 7](docs/phase7_methodology.md).
- [Metodologia da Fase 8](docs/phase8_methodology.md).

## Reprodutibilidade e limitações

- O RUL absoluto é alvo supervisionado e nunca é utilizado como atributo de entrada.
- Divisões, incerteza e comparações estatísticas tratam o rolamento como unidade experimental.
- O XJTU-SY não fornece um rótulo oficial para o instante exato de início da degradação.
- Os endpoints representam o término experimental run-to-failure documentado, sem pressupor
  destruição física.
- Cenários de horizonte fixo usam censura administrativa para avaliação operacional.
- Não há validação externa ou implantação industrial.

As definições formais, políticas de causalidade, resultados por fase e limitações completas
estão no [guia técnico](docs/technical_guide.md).
