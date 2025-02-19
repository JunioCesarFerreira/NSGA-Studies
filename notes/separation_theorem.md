# Teorema de Separação de Conjuntos Convexos

O **Teorema de Separação de Conjuntos Convexos** afirma que:

Dado dois conjuntos convexos não vazios $A$ e $B$ em um espaço vetorial normado (ou mais geralmente, em um espaço vetorial topológico), sob certas condições, existe um hiperplano que separa $A$ e $B$. O tipo de separação depende da posição relativa entre $A$ e $B$.

### Enunciado formal (caso básico em \(\mathbb{R}^n\)):

Sejam $A$ e $B$ dois conjuntos convexos não vazios em \(\mathbb{R}^n\) e $A \cap B = \emptyset$. Então:

1. **Separação fraca**: Se $A$ e $B$ são disjuntos, existe um vetor $w \in \mathbb{R}^n \setminus \{0\}$ e um escalar $\alpha \in \mathbb{R}$ tais que:  
$$
\sup_{x \in A} w \cdot x \leq \alpha \leq \inf_{y \in B} w \cdot y,
$$  
   com pelo menos uma desigualdade estrita, garantindo que o hiperplano definido por $w \cdot x = \alpha$ separa $A$ de $B$.

2. **Separação forte**: Se $A$ e $B$ são convexos fechados e disjuntos, então existe $w \in \mathbb{R}^n \setminus \{0\}$ e $\alpha_1, \alpha_2 \in \mathbb{R}$ com $\alpha_1 < \alpha_2$ tais que:  
$$
w \cdot x \leq \alpha_1, \quad \forall x \in A \quad \text{e} \quad w \cdot y \geq \alpha_2, \quad \forall y \in B.
$$  

### Intuição:
Este teorema é fundamental na geometria convexa, pois garante que conjuntos convexos disjuntos podem ser separados por hiperplanos. Em espaços normados mais gerais, variantes desse teorema requerem propriedades adicionais como a convexidade fechada ou a compacidade de um dos conjuntos.

Ref.: "Convex Analysis" - R. Tyrrell Rockafellar

---

# Aplicação do Teorema no contexto da fronteira de Pareto

Vamos analisar o problema geometricamente e usar o **Teorema de Separação de Conjuntos Convexos** para obter uma condição explícita para a separação entre um paralelepípedo $P$ $n$-dimensional e um segmento de reta $R$ em $\mathbb{R}^n$.


### 1. Caracterização geométrica do paralelepípedo $P$:

Suponha que o paralelepípedo $P$ seja definido como:  
$$
P = \{ x \in \mathbb{R}^n \mid a_i \leq x_i \leq b_i, \, i = 1, 2, \dots, n \},
$$
onde $a = (a_1, a_2, \dots, a_n)$ e $b = (b_1, b_2, \dots, b_n)$ são os vértices opostos que determinam $P$. Aqui, $a_i$ é o menor valor para a $i$-ésima coordenada e $b_i$ é o maior.


### 2. Caracterização geométrica do segmento $R$:

Seja $R$ um segmento de reta definido pelos pontos $p_1$ e $p_2$ em $\mathbb{R}^n$:  
$$
R = \{ p(t) = (1 - t) p_1 + t p_2 \mid t \in [0, 1] \},
$$
onde $p_1 = (p_{1,1}, p_{1,2}, \dots, p_{1,n})$ e $p_2 = (p_{2,1}, p_{2,2}, \dots, p_{2,n})$ são os extremos do segmento.


### 3. Condição para separação usando o teorema de separação:

O Teorema de Separação de Conjuntos Convexos afirma que $P \cap R = \emptyset$ se, e somente se, existe uma desigualdade linear que separa os dois conjuntos. Em termos geométricos, para $P \cap R = \emptyset$, deve ocorrer o seguinte:

Para cada coordenada $i$, o intervalo $[a_i, b_i]$ definido pelas extremidades de $P$ **não pode sobrepor** o intervalo gerado pelas projeções de $p_1$ e $p_2$ na mesma coordenada $i$. Em outras palavras:

1. Calcule as projeções do segmento $R$ em cada dimensão $i$:  
   $$
   [\min(p_{1,i}, p_{2,i}), \max(p_{1,i}, p_{2,i})].
   $$

2. O segmento $R$ está **fora** de $P$ se, e somente se, **para alguma dimensão $i$**, o intervalo $[\min(p_{1,i}, p_{2,i}), \max(p_{1,i}, p_{2,i})]$ não sobrepõe o intervalo $[a_i, b_i]$. Formalmente:  
   $$
   \max(p_{1,i}, p_{2,i}) < a_i \quad \text{ou} \quad \min(p_{1,i}, p_{2,i}) > b_i,
   $$
   para algum $i \in \{1, 2, \dots, n\}$.


### 4. Interpretação das condições:

Portanto, $R \cap P = \emptyset$ se, e somente se, existe pelo menos uma dimensão $i$ tal que o segmento $R$ está **completamente à esquerda ou à direita** do paralelepípedo em $i$. Isso implica:

- **Condição para interseção vazia** ($R \cap P = \emptyset$):  
  $$
  \exists i \in \{1, 2, \dots, n\}, \, \max(p_{1,i}, p_{2,i}) < a_i \quad \text{ou} \quad \min(p_{1,i}, p_{2,i}) > b_i.
  $$

- **Condição para interseção não vazia** ($R \cap P \neq \emptyset$):  
  Para toda dimensão $i$, deve haver sobreposição:
  $$
  \min(p_{1,i}, p_{2,i}) \leq b_i \quad \text{e} \quad \max(p_{1,i}, p_{2,i}) \geq a_i.
  $$


### 5. Conexão com o Teorema de Separação:
O teorema garante que se $R \cap P = \emptyset$, então existe um hiperplano que separa os conjuntos. No caso geométrico de um paralelepípedo $P$ e um segmento $R$, os **hiperplanos de separação** correspondem aos limites $x_i = a_i$ ou $x_i = b_i$ de $P$. As desigualdades acima expressam como os extremos de $R$ se comportam em relação a essas fronteiras.

---

## Referências sobre o assunto


**Livros introdutórios (Geometria Convexa e Análise Funcional):**

1. **"Convex Analysis" - R. Tyrrell Rockafellar**  
   - Este é um clássico sobre análise convexa, cobrindo o teorema de separação de maneira rigorosa e abrangente.  
   - Recomendado para quem quer uma base sólida em geometria convexa e suas conexões com otimização.  

2. **"Functional Analysis" - Walter Rudin**  
   - Este livro apresenta o teorema de separação no contexto de espaços vetoriais normados e análise funcional.  
   - Ideal para um leitor com conhecimento intermediário-avançado em análise matemática.


**Livros intermediários (Matemática Aplicada e Otimização):**

3. **"Convex Optimization" - Stephen Boyd e Lieven Vandenberghe**  
   - Este livro é mais aplicado, com foco em otimização convexa. Ele cobre o teorema de separação como parte das ferramentas para estudar dualidade e otimização.  
   - Disponível gratuitamente no site do autor: [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).

4. **"Introductory Functional Analysis with Applications" - Erwin Kreyszig**  
   - Uma introdução clara à análise funcional, com o teorema de separação no contexto de espaços normados.  
   - Mais acessível que o livro de Rudin, mas ainda rigoroso.

**Referências avançadas:**

5. **"The Geometry of the Simplex Method" - Karl-Heinz Borgwardt**  
   - Explora aspectos geométricos da separação convexa no contexto de otimização linear.

6. **"Linear and Nonlinear Functional Analysis with Applications" - Philippe G. Ciarlet**  
   - Aborda o teorema de separação no contexto geral de espaços vetoriais topológicos e inclui muitos exemplos aplicados.


**Artigos e materiais gratuitos:**

7. **"Notes on Convex Sets and Functions" - Dimitri Bertsekas**  
   - Notas concisas e gratuitas que incluem o teorema de separação e suas aplicações.  
   - Disponível no site do autor: [Dimitri Bertsekas Publications](http://www.mit.edu/~dimitrib/).
