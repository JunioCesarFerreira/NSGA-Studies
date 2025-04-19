# Estudos sobre NSGA

Este repositório fornece material didático de introdução a problemas de otimização multiobjetivo, com foco na família Non-dominated Sorting Genetic Algorithm (NSGA). Incluindo explicações, implementações e exemplos para ajudar a entender os princípios e aplicações dos algoritmos NSGA.

## Catálogo

### Estudo Inicial

Iniciamos os estudos apresentando alguns conceitos básicos e utilizando um exemplo simples para ilustrar a teoria. Os notebooks listados a seguir são desenvolvidos com base nesse exemplo.

#### Notebooks

**Básico**

- [Estudo Inicial](./notebooks/basic/nsga-initial-study.ipynb)

- [Implementação NSGA-II sem crowding distance](./notebooks/basic/nsga-1.ipynb)

- [Implementação NSGA-II](./notebooks/basic/nsga-2.ipynb)

- [Implementação NSGA-III](./notebooks/basic/nsga-3.ipynb)

- [Exemplo padrão DEAP](./notebooks/basic/deap-example.ipynb)

- [Teste com bibliotecas DEAP e pymoo](./notebooks/basic/nsga-lib-study.ipynb)

- [Visulização dos pontos de referências em simplexos](./notebooks/basic/test-simplex-dist.ipynb)

**Comparações**

- [Comparação de resultados das diferentes implementações](./notebooks/comparative/nsga-comparations.ipynb)

- [Refinamento da comparação de resultados das diferentes implementações](./notebooks/comparative/nsga-comparations2.ipynb)

- [Exemplos Ilustrativos em duas dimensões](./notebooks/comparative/test-2d-analysis.ipynb)

**Implementações Genéricas**

- [Implementações genéricas multidimensionais do NSGA-II e NSGA-III](./notebooks/generic/multidimensional-nsga.ipynb)

**Problemas**

- Exemplo de aplicação para posicionamento de motes em rede RPL
    - [Abordagem Inicial](./notebooks/problem/rpl-dodag/1-rpl-dodag-nsga.ipynb)
    - [Melhoria no DODAG e na população inicial](./notebooks/problem/rpl-dodag/2-rpl-dodag-nsga.ipynb)
    - [Estudos de Crossover em DODAG](./notebooks/problem/rpl-dodag/digraph-crossover.ipynb)

**Próximos Estudos**

- Biblioteca DEAP
    - [algorithms.py](https://github.com/DEAP/deap/blob/master/deap/algorithms.py)
    - [selNSGA3](https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L492)

- Biblioteca Pymoo
    - [nsga3.py](https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/nsga3.py)

## Ferramentas Utilizadas

Os estudos deste repositório são conduzidos por meio de **notebooks Jupyter**, utilizando o **Visual Studio Code (VS Code)** como ambiente de desenvolvimento integrado (IDE). Para a execução dos códigos, é usada a versão **Python 3.12.1**. Certifique-se de ter essas ferramentas configuradas para reproduzir e executar os exemplos localmente.

## **Alerta**

Todo o material deste repositório é destinado a estudos e pesquisas sobre NSGA. Por isso, pode conter erros. Caso identifique algum, sua contribuição será muito bem-vinda. Não hesite em entrar em contato!

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

