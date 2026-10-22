# 📚 Enciclopedia de Algoritmos

> De la búsqueda lineal al Transformer. Cada entrada incluye **idea**, **base matemática**, **pseudocódigo** y **complejidad**, con el detalle suficiente para implementarla desde cero sin mirar código ajeno.

## Cómo leer esta enciclopedia

**Nivel de dificultad:** ⭐ básico · ⭐⭐ intermedio · ⭐⭐⭐ avanzado

**Plantilla de cada entrada:**

| Campo | Contenido |
|---|---|
| **Problema** | Qué resuelve y cuándo se usa |
| **Idea** | La intuición en 2-3 líneas |
| **Matemáticas** | Fórmulas, invariantes, demostraciones clave |
| **Pseudocódigo** | Lo mínimo para codificarlo |
| **Complejidad** | Tiempo / espacio |
| **Trampas** | Errores típicos al implementar |

**Convenciones:** índices desde 0 · `n` = tamaño de entrada · `V`,`E` = vértices y aristas · `⊙` = producto elemento a elemento · `‖·‖` = norma euclídea · `∇` = gradiente. Las fórmulas usan LaTeX (GitHub las renderiza con `$...$` y `$$...$$`).

## Índice

1. [Fundamentos matemáticos](#1-fundamentos-matemáticos)
2. [Búsqueda](#2-búsqueda)
3. [Ordenación](#3-ordenación)
4. [Estructuras de datos algorítmicas](#4-estructuras-de-datos-algorítmicas)
5. [Grafos](#5-grafos)
6. [Programación dinámica](#6-programación-dinámica)
7. [Cadenas de texto](#7-cadenas-de-texto)
8. [Algoritmos numéricos y teoría de números](#8-algoritmos-numéricos-y-teoría-de-números)
9. [Geometría computacional](#9-geometría-computacional)
10. [Compresión, criptografía y códigos](#10-compresión-criptografía-y-códigos)
11. [Aleatoriedad y estructuras probabilísticas](#11-aleatoriedad-y-estructuras-probabilísticas)
12. [Backtracking, juegos y búsqueda en IA clásica](#12-backtracking-juegos-y-búsqueda-en-ia-clásica)
13. [Optimización continua](#13-optimización-continua)
14. [Metaheurísticas y algoritmos evolutivos](#14-metaheurísticas-y-algoritmos-evolutivos)
15. [Machine Learning supervisado](#15-machine-learning-supervisado)
16. [Machine Learning no supervisado](#16-machine-learning-no-supervisado)
17. [Modelos probabilísticos y secuenciales](#17-modelos-probabilísticos-y-secuenciales)
18. [Aprendizaje por refuerzo](#18-aprendizaje-por-refuerzo)
19. [Deep Learning](#19-deep-learning)
20. [Hoja de ruta de implementación](#20-hoja-de-ruta-de-implementación)

---

# 1. Fundamentos matemáticos

## 1.1 Notación asintótica
- $f = O(g)$: existen $c, n_0$ con $f(n) \le c\,g(n)$ para $n \ge n_0$ (cota superior).
- $f = \Omega(g)$: $f(n) \ge c\,g(n)$ (cota inferior). $f = \Theta(g)$: ambas a la vez.
- Jerarquía habitual: $1 < \log n < \sqrt n < n < n\log n < n^2 < n^3 < 2^n < n!$

## 1.2 Teorema maestro
Para $T(n) = a\,T(n/b) + f(n)$, con $a\ge1, b>1$. Sea $c^* = \log_b a$:

| Caso | Condición | Resultado |
|---|---|---|
| 1 | $f(n) = O(n^{c^*-\varepsilon})$ | $T(n)=\Theta(n^{c^*})$ |
| 2 | $f(n) = \Theta(n^{c^*}\log^k n)$ | $T(n)=\Theta(n^{c^*}\log^{k+1} n)$ |
| 3 | $f(n) = \Omega(n^{c^*+\varepsilon})$ y $a f(n/b)\le c f(n)$, $c<1$ | $T(n)=\Theta(f(n))$ |

Ejemplo: mergesort, $a=2,b=2,f=n$ → caso 2 → $\Theta(n\log n)$.

## 1.3 Álgebra lineal imprescindible
- Producto matriz-vector: $(Ax)_i = \sum_j A_{ij}x_j$. Producto matricial: $(AB)_{ij}=\sum_k A_{ik}B_{kj}$.
- Norma $L_2$: $\|x\|=\sqrt{\sum x_i^2}$. Producto escalar: $x\cdot y=\|x\|\|y\|\cos\theta$.
- Autovalores: $Av=\lambda v$. Descomposición SVD: $A=U\Sigma V^\top$.
- Gradiente de formas comunes: $\nabla_x(a^\top x)=a$, $\nabla_x\|Ax-b\|^2=2A^\top(Ax-b)$.

## 1.4 Cálculo y regla de la cadena
Si $y=f(g(x))$ entonces $\dfrac{dy}{dx}=f'(g(x))\,g'(x)$. En varias variables: $\dfrac{\partial L}{\partial x}=\sum_k \dfrac{\partial L}{\partial u_k}\dfrac{\partial u_k}{\partial x}$. **Es la base de la retropropagación** (cap. 19).

## 1.5 Probabilidad básica
- Bayes: $P(A|B)=\dfrac{P(B|A)P(A)}{P(B)}$.
- Esperanza: $E[X]=\sum x\,p(x)$; varianza: $\mathrm{Var}(X)=E[X^2]-E[X]^2$.
- Gaussiana: $\mathcal N(x|\mu,\sigma^2)=\dfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$.
- Entropía: $H(p)=-\sum p_i\log p_i$. KL: $D_{KL}(p\|q)=\sum p_i\log\frac{p_i}{q_i}\ge0$.

---

# 2. Búsqueda

## 2.1 Búsqueda lineal ⭐
**Problema:** encontrar `x` en una colección sin orden.
**Idea:** recorrer elemento a elemento.
```
for i in 0..n-1: if A[i]==x: return i
return -1
```
**Complejidad:** $O(n)$ tiempo, $O(1)$ espacio. Con centinela se ahorra una comparación por iteración.

## 2.2 Búsqueda binaria ⭐
**Problema:** buscar en array **ordenado**.
**Idea:** comparar con el medio y descartar la mitad.
**Matemáticas:** $T(n)=T(n/2)+1\Rightarrow T(n)=\lfloor\log_2 n\rfloor+1$ comparaciones.
```
lo=0; hi=n-1
while lo<=hi:
    mid = lo + (hi-lo)//2        # evita overflow
    if A[mid]==x: return mid
    if A[mid]<x: lo=mid+1 else: hi=mid-1
return -1
```
**Trampas:** `(lo+hi)/2` desborda en enteros de 32 bits; bucle infinito si no haces `mid±1`. Variantes útiles: *lower_bound* (primer índice con `A[i] >= x`) y *upper_bound*. También sirve para "buscar la respuesta" en problemas monótonos (búsqueda binaria sobre el resultado).

## 2.3 Búsqueda por interpolación ⭐⭐
**Idea:** estimar la posición como lo haría un humano en una guía telefónica.
**Matemáticas:** interpolación lineal
$$pos = lo + \frac{(x-A[lo])\,(hi-lo)}{A[hi]-A[lo]}$$
**Complejidad:** $O(\log\log n)$ si los datos son uniformes; $O(n)$ en el peor caso.
**Trampas:** división por cero si $A[hi]=A[lo]$.

## 2.4 Búsqueda exponencial ⭐⭐
**Idea:** para listas "infinitas" o cuando `x` está cerca del inicio: duplicar `i=1,2,4,8…` hasta que `A[i] ≥ x`, y luego búsqueda binaria en `[i/2, i]`.
**Complejidad:** $O(\log p)$, con $p$ la posición de `x`.

## 2.5 Búsqueda por saltos (Jump Search) ⭐⭐
**Idea:** saltar bloques de tamaño $m$ y luego lineal dentro del bloque.
**Matemáticas:** coste $\frac{n}{m}+m$, mínimo en $m=\sqrt n$ → $O(\sqrt n)$.

## 2.6 Búsqueda ternaria y sección áurea ⭐⭐
**Problema:** máximo de una función **unimodal** $f$ en $[l,r]$.
```
while r-l > eps:
    m1 = l+(r-l)/3; m2 = r-(r-l)/3
    if f(m1) < f(m2): l = m1 else: r = m2
```
**Matemáticas:** cada iteración reduce el intervalo a $2/3$. La **sección áurea** usa $\varphi=\frac{1+\sqrt5}{2}$ y reutiliza una evaluación por iteración (reducción $1/\varphi\approx0.618$).

---

# 3. Ordenación

**Cota inferior:** cualquier ordenación por comparaciones necesita $\ge\log_2(n!)=\Theta(n\log n)$ comparaciones (el árbol de decisión tiene $n!$ hojas y altura $\ge\log_2 n!$).

| Algoritmo | Mejor | Medio | Peor | Espacio | Estable |
|---|---|---|---|---|---|
| Burbuja | $n$ | $n^2$ | $n^2$ | 1 | Sí |
| Selección | $n^2$ | $n^2$ | $n^2$ | 1 | No |
| Inserción | $n$ | $n^2$ | $n^2$ | 1 | Sí |
| Shell | $n\log n$ | ~$n^{1.3}$ | $n^2$ | 1 | No |
| Merge | $n\log n$ | $n\log n$ | $n\log n$ | $n$ | Sí |
| Quick | $n\log n$ | $n\log n$ | $n^2$ | $\log n$ | No |
| Heap | $n\log n$ | $n\log n$ | $n\log n$ | 1 | No |
| Counting | $n+k$ | $n+k$ | $n+k$ | $k$ | Sí |
| Radix | $d(n+k)$ | $d(n+k)$ | $d(n+k)$ | $n+k$ | Sí |
| Bucket | $n$ | $n$ | $n^2$ | $n$ | Sí |

## 3.1 Burbuja ⭐
Comparar vecinos e intercambiar si están desordenados; tras la pasada $i$, los $i$ mayores están en su sitio.
```
for i in 0..n-2:
    swapped=false
    for j in 0..n-2-i:
        if A[j]>A[j+1]: swap; swapped=true
    if not swapped: break
```

## 3.2 Selección ⭐
En cada paso buscar el mínimo del resto y colocarlo en posición $i$. Siempre $\frac{n(n-1)}{2}$ comparaciones, pero solo $n-1$ intercambios.

## 3.3 Inserción ⭐
Mantener un prefijo ordenado e insertar cada nuevo elemento desplazando mayores a la derecha.
```
for i in 1..n-1:
    key=A[i]; j=i-1
    while j>=0 and A[j]>key: A[j+1]=A[j]; j--
    A[j+1]=key
```
Excelente para arrays pequeños o casi ordenados ($O(n+\text{inversiones})$). Se usa dentro de Timsort y Quicksort híbridos.

## 3.4 Shell Sort ⭐⭐
Inserción con saltos decrecientes $h$ (p. ej. secuencia de Knuth $h=3h+1$: 1,4,13,40…). Cada pasada deja el array "$h$-ordenado"; la última ($h=1$) es una inserción sobre datos casi ordenados.

## 3.5 Merge Sort ⭐⭐
**Idea:** dividir en dos, ordenar recursivamente, mezclar.
**Matemáticas:** $T(n)=2T(n/2)+\Theta(n)\Rightarrow\Theta(n\log n)$.
```
merge_sort(A):
    if len(A)<=1: return A
    L=merge_sort(A[:m]); R=merge_sort(A[m:])
    i=j=0; out=[]
    while i<len(L) and j<len(R):
        if L[i]<=R[j]: out.append(L[i]); i++    # <= garantiza estabilidad
        else: out.append(R[j]); j++
    return out + L[i:] + R[j:]
```

## 3.6 Quick Sort ⭐⭐
**Idea:** elegir pivote, particionar (menores | pivote | mayores), recursión en ambos lados.
**Matemáticas:** con pivote aleatorio el número esperado de comparaciones es $2n\ln n\approx1.39\,n\log_2 n$.
```
partition(A,lo,hi):            # Lomuto
    p=A[hi]; i=lo
    for j in lo..hi-1:
        if A[j]<p: swap(A[i],A[j]); i++
    swap(A[i],A[hi]); return i
```
**Trampas:** peor caso $O(n^2)$ con datos ordenados y pivote fijo; usa pivote aleatorio o mediana de 3. Recurre primero en la mitad más pequeña para limitar la pila a $O(\log n)$. Con muchos duplicados usa partición de 3 vías (Dutch flag).

## 3.7 Heap Sort ⭐⭐
1) Construir un *max-heap* en $O(n)$; 2) intercambiar raíz con el último y hacer *sift-down*.
**Matemáticas:** índices: padre $\lfloor(i-1)/2\rfloor$, hijos $2i+1,2i+2$. Construir heap es $O(n)$ porque $\sum_h \frac{n}{2^{h+1}}\,h=O(n)$.

## 3.8 Counting Sort ⭐⭐
Para enteros en $[0,k)$.
```
count[x]++ para cada x
count[i] += count[i-1]                  # sumas prefijas = posición final
for x in reversed(A): out[--count[x]] = x   # reversed => estable
```

## 3.9 Radix Sort (LSD) ⭐⭐
Ordenar por dígito menos significativo a más, usando un orden **estable** (counting) en cada pasada. $d$ dígitos en base $b$ → $O(d(n+b))$.

## 3.10 Bucket Sort ⭐⭐
Repartir en $n$ cubos según el valor (datos ~uniformes en $[0,1)$: `bucket = floor(n*x)`), ordenar cada cubo con inserción y concatenar. Esperado $O(n)$.

## 3.11 Timsort ⭐⭐⭐
Híbrido de merge + inserción: detecta *runs* ya ordenados (los alarga a un mínimo `minrun` con inserción binaria) y los fusiona manteniendo invariantes de pila sobre sus longitudes; usa *galloping* para fusiones desbalanceadas. Es el sort de Python y Java (objetos). $O(n)$ en el mejor caso, $O(n\log n)$ en el peor.

---

# 4. Estructuras de datos algorítmicas

## 4.1 Pila, cola y deque ⭐
LIFO/FIFO/doble extremo. Con array circular o lista enlazada, todas las operaciones son $O(1)$. Una **pila monótona** resuelve "siguiente elemento mayor" en $O(n)$.

## 4.2 Montículo binario (heap) ⭐⭐
Árbol completo en array con propiedad $A[parent]\ge A[child]$.
```
sift_up(i):   while i>0 and A[p(i)]<A[i]: swap; i=p(i)
sift_down(i): loop: m=argmax(A[i],A[2i+1],A[2i+2]); if m==i break; swap; i=m
push: append + sift_up           pop: swap(0,last); remove; sift_down(0)
```
`push`/`pop` en $O(\log n)$, `peek` en $O(1)$. Base de colas de prioridad (Dijkstra, Prim, Huffman).

## 4.3 Árbol binario de búsqueda (BST) ⭐⭐
Para cada nodo: izquierda < nodo < derecha. Buscar/insertar $O(h)$. Con claves aleatorias $h\approx 2\ln n$; con claves ordenadas degenera a $h=n$. Borrado: 0 hijos (quitar), 1 hijo (puentear), 2 hijos (sustituir por el sucesor inorden).

## 4.4 Árbol AVL ⭐⭐⭐
BST autobalanceado: factor de balance $bf=h(L)-h(R)\in\{-1,0,1\}$. Tras insertar/borrar, se corrige con **rotaciones** (simple izq/der, doble izq-der/der-izq).
**Matemáticas:** altura $h\le1.44\log_2(n+2)$ (los árboles de Fibonacci son los más desbalanceados). Alternativa: rojo-negro (menos rotaciones, altura $\le2\log_2(n+1)$).

## 4.5 Tabla hash ⭐⭐
**Idea:** $h(k)\in[0,m)$ indexa un array.
**Matemáticas:** familia universal $h(k)=((ak+b)\bmod p)\bmod m$, con $p$ primo $>$ universo, $a\in[1,p),b\in[0,p)$. Con factor de carga $\alpha=n/m$, en *chaining* el coste esperado es $1+\alpha$.
**Resolución de colisiones:** encadenamiento, sondeo lineal $h(k)+i$, cuadrático, doble hash $h_1+i\,h_2$, *cuckoo hashing*. Redimensionar (×2) cuando $\alpha$ supera ~0.7 → coste amortizado $O(1)$.

## 4.6 Trie (árbol de prefijos) ⭐⭐
Cada nodo tiene hijos por carácter y un flag `fin`. Insertar/buscar en $O(L)$ ($L$=longitud de la palabra). Autocompletado: bajar al prefijo y hacer DFS.

## 4.7 Union-Find (conjuntos disjuntos) ⭐⭐
```
find(x): if parent[x]!=x: parent[x]=find(parent[x])   # compresión de caminos
union(a,b): ra,rb=find(a),find(b); si distinto: unir por rango/tamaño
```
Con ambas optimizaciones: $O(\alpha(n))$ amortizado, donde $\alpha$ es la inversa de Ackermann (≤4 en la práctica). Usado en Kruskal, componentes conexas, percolación.

## 4.8 Árbol de Fenwick (BIT) ⭐⭐
Sumas prefijas con actualizaciones puntuales en $O(\log n)$ y casi sin memoria extra.
```
update(i,d): while i<=n: T[i]+=d; i += i & -i
query(i):    s=0; while i>0: s+=T[i]; i -= i & -i; return s      # suma de 1..i
```
Intuición: `T[i]` guarda la suma de las `i & -i` posiciones que terminan en `i` (indexación desde 1).

## 4.9 Segment Tree ⭐⭐⭐
Árbol donde cada nodo almacena una operación asociativa (suma, min, gcd…) del rango que cubre.
- Consulta de rango y actualización puntual: $O(\log n)$. Memoria $4n$.
- **Propagación perezosa (lazy):** actualizaciones de rango en $O(\log n)$, guardando una "etiqueta pendiente" que se empuja a los hijos al descender.

## 4.10 Sparse Table ⭐⭐
Para consultas de mínimo/máximo **sin actualizaciones**: $st[k][i]=\min(st[k-1][i],\,st[k-1][i+2^{k-1}])$. Consulta $[l,r]$ en $O(1)$: $k=\lfloor\log_2(r-l+1)\rfloor$, $\min(st[k][l],st[k][r-2^k+1])$. Preproceso $O(n\log n)$.

## 4.11 Lista de saltos (Skip List) ⭐⭐⭐
Lista enlazada con niveles; cada nodo se promociona al nivel superior con probabilidad $p=1/2$. Búsqueda/inserción esperadas $O(\log n)$. Alternativa simple a los árboles balanceados.

---

# 5. Grafos

**Representaciones:** lista de adyacencia (memoria $O(V+E)$, la habitual), matriz de adyacencia ($O(V^2)$, consulta de arista en $O(1)$), lista de aristas (para Kruskal/Bellman-Ford).

## 5.1 BFS (búsqueda en anchura) ⭐
**Problema:** camino más corto en grafos **sin pesos** y recorrido por niveles.
```
dist[s]=0; Q=[s]
while Q: u=pop_front(Q)
    for v in adj[u]: if dist[v]==inf: dist[v]=dist[u]+1; push(Q,v)
```
$O(V+E)$. Sirve para comprobar bipartición (colorear alternando) y laberintos. **0-1 BFS:** deque, pesos 0 al frente y 1 al final.

## 5.2 DFS (búsqueda en profundidad) ⭐
Explorar hasta el fondo y retroceder. Con tiempos de entrada/salida `tin/tout` clasifica aristas (árbol, retroceso, cruce). Base de: detección de ciclos, orden topológico, componentes fuertemente conexas, puentes. $O(V+E)$. Cuidado con el desbordamiento de pila en grafos profundos → versión iterativa.

## 5.3 Dijkstra ⭐⭐
**Problema:** caminos mínimos desde un origen con pesos **no negativos**.
**Matemáticas:** relajación: si $d[v]>d[u]+w(u,v)$ entonces $d[v]\leftarrow d[u]+w(u,v)$. Correcto porque el nodo extraído con menor $d$ ya es óptimo (no hay pesos negativos que lo mejoren).
```
d[s]=0; PQ={(0,s)}
while PQ: (du,u)=pop_min(PQ); if du>d[u]: continue
    for (v,w) in adj[u]: if d[u]+w<d[v]: d[v]=d[u]+w; push(PQ,(d[v],v)); prev[v]=u
```
$O((V+E)\log V)$ con heap binario. **Trampa:** falla con pesos negativos.

## 5.4 Bellman-Ford ⭐⭐
Relaja **todas** las aristas $V-1$ veces; una iteración más que aún mejore ⇒ hay ciclo negativo alcanzable. $O(VE)$. Con cola de nodos modificados (SPFA) es más rápido en la práctica.

## 5.5 Floyd-Warshall ⭐⭐
**Problema:** distancias entre **todos** los pares.
**Matemáticas:** $d^{(k)}_{ij}=\min\big(d^{(k-1)}_{ij},\ d^{(k-1)}_{ik}+d^{(k-1)}_{kj}\big)$, usando solo los nodos $\{1..k\}$ como intermedios.
```
for k: for i: for j: d[i][j]=min(d[i][j], d[i][k]+d[k][j])    # k SIEMPRE el bucle externo
```
$O(V^3)$. Ciclo negativo si algún $d[i][i]<0$. Con operaciones (∨,∧) calcula la clausura transitiva (Warshall).

## 5.6 Johnson ⭐⭐⭐
Todos los pares en grafos dispersos con pesos negativos: ejecutar Bellman-Ford desde un nodo virtual para obtener potenciales $h(v)$, reponderar $w'(u,v)=w(u,v)+h(u)-h(v)\ge0$ y lanzar Dijkstra desde cada nodo. $O(V E\log V)$.

## 5.7 A* ⭐⭐
Dijkstra guiado por heurística: ordena por $f(n)=g(n)+h(n)$.
**Matemáticas:** si $h$ es **admisible** ($h\le$ coste real) A* es óptimo; si además es **consistente** ($h(u)\le w(u,v)+h(v)$) no necesita reabrir nodos. Heurísticas típicas: Manhattan, euclídea, octil.

## 5.8 Árbol de expansión mínimo: Kruskal ⭐⭐
Ordenar aristas por peso y añadir la siguiente si no forma ciclo (Union-Find). $O(E\log E)$.
**Propiedad de corte:** la arista más ligera que cruza cualquier corte pertenece a algún AEM.

## 5.9 Prim ⭐⭐
Crecer un árbol desde un nodo eligiendo siempre la arista más barata que sale del árbol (heap). $O(E\log V)$. Mejor en grafos densos.

## 5.10 Orden topológico (Kahn) ⭐⭐
Para DAGs. Calcular grados de entrada; sacar nodos con grado 0, "borrar" sus aristas y repetir. Si quedan nodos ⇒ hay ciclo. $O(V+E)$. Alternativa: DFS y orden inverso de salida.

## 5.11 Componentes fuertemente conexas (Tarjan / Kosaraju) ⭐⭐⭐
**Kosaraju:** DFS para obtener orden de salida → DFS sobre el grafo **transpuesto** en orden inverso; cada árbol es una SCC.
**Tarjan:** una sola DFS con `low[u]=min(tin[u], low[v] de hijos, tin[w] de aristas a pila)`; si `low[u]==tin[u]`, u es raíz de una SCC y se extrae de la pila. $O(V+E)$.

## 5.12 Puntos de articulación y puentes ⭐⭐⭐
DFS con `low`. Arista $(u,v)$ (hijo $v$) es **puente** si $low[v]>tin[u]$. Nodo $u$ (no raíz) es de articulación si algún hijo cumple $low[v]\ge tin[u]$; la raíz, si tiene $\ge2$ hijos.

## 5.13 Flujo máximo: Edmonds-Karp / Dinic ⭐⭐⭐
**Teorema max-flow min-cut:** el flujo máximo = capacidad del corte mínimo.
**Ford-Fulkerson:** buscar camino aumentante en el **grafo residual** ($c_f(u,v)=c(u,v)-f(u,v)$, y arista inversa con $f(u,v)$), enviar el cuello de botella y repetir.
**Edmonds-Karp:** caminos con BFS → $O(VE^2)$. **Dinic:** BFS construye el grafo de niveles + DFS con "flujo bloqueante" → $O(V^2E)$; $O(E\sqrt V)$ en emparejamientos unitarios.

## 5.14 Emparejamiento bipartito: Hopcroft-Karp y Húngaro ⭐⭐⭐
**Hopcroft-Karp:** fases de BFS (niveles) + DFS de caminos aumentantes disjuntos, $O(E\sqrt V)$.
**Húngaro:** asignación de coste mínimo, $O(n^3)$, con potenciales (duales) sobre filas y columnas para mantener costes reducidos $\ge0$.

## 5.15 LCA (ancestro común más bajo) ⭐⭐
*Binary lifting:* `up[k][v]` = ancestro $2^k$; subir al más profundo y luego ambos a la vez. Preproceso $O(n\log n)$, consulta $O(\log n)$.

## 5.16 PageRank ⭐⭐
**Matemáticas:** $r=d\,M\,r+\frac{1-d}{N}\mathbf 1$, con $M_{ij}=1/\text{outdeg}(j)$ si $j\to i$, y $d\approx0.85$. Se resuelve por **iteración de la potencia** hasta que $\|r_{t+1}-r_t\|_1<\varepsilon$. Los nodos sin salidas ("sumideros") se reparten uniformemente.

---

# 6. Programación dinámica

**Receta:** (1) definir el estado, (2) escribir la recurrencia, (3) casos base, (4) orden de cálculo (o memoización). Requiere **subestructura óptima** y **subproblemas solapados**.

## 6.1 Fibonacci ⭐
$F_n=F_{n-1}+F_{n-2}$. Memoización $O(n)$, o exponenciación de matriz en $O(\log n)$:
$$\begin{pmatrix}F_{n+1}\\F_n\end{pmatrix}=\begin{pmatrix}1&1\\1&0\end{pmatrix}^n\begin{pmatrix}1\\0\end{pmatrix}$$

## 6.2 Mochila 0/1 ⭐⭐
**Estado:** `dp[i][w]` = mejor valor con los primeros $i$ objetos y capacidad $w$.
$$dp[i][w]=\max\big(dp[i-1][w],\ dp[i-1][w-w_i]+v_i\big)\quad(w\ge w_i)$$
$O(nW)$ (pseudopolinómico). En 1D recorre $w$ **de mayor a menor**. Mochila no acotada: recorre de menor a mayor.

## 6.3 Cambio de monedas ⭐⭐
Mínimo de monedas: $dp[x]=1+\min_c dp[x-c]$, $dp[0]=0$. Número de formas: bucle externo sobre monedas, interno sobre $x$: $dp[x]\mathrel{+}=dp[x-c]$ (el orden evita contar permutaciones).

## 6.4 Subsecuencia común más larga (LCS) ⭐⭐
$$dp[i][j]=\begin{cases}dp[i-1][j-1]+1 & a_i=b_j\\ \max(dp[i-1][j],dp[i][j-1]) & \text{si no}\end{cases}$$
$O(nm)$. Reconstruir retrocediendo desde $dp[n][m]$. Es la base de `diff`.

## 6.5 Distancia de edición (Levenshtein) ⭐⭐
$$d[i][j]=\min\begin{cases}d[i-1][j]+1&\text{borrar}\\ d[i][j-1]+1&\text{insertar}\\ d[i-1][j-1]+[a_i\ne b_j]&\text{sustituir}\end{cases}$$
Casos base $d[i][0]=i$, $d[0][j]=j$. **Needleman-Wunsch** (alineamiento global en bioinformática) es lo mismo con matriz de puntuación y penalización de huecos; **Smith-Waterman** (local) añade $\max(0,\cdot)$.

## 6.6 Subsecuencia creciente más larga (LIS) ⭐⭐
$O(n^2)$ con `dp[i]=1+max dp[j]` ($j<i$, $a_j<a_i$). En $O(n\log n)$: mantener `tails[k]` = menor último elemento de una subsecuencia creciente de longitud $k+1$; para cada $x$, búsqueda binaria (*lower_bound*) y reemplazo.

## 6.7 Multiplicación de cadena de matrices ⭐⭐⭐
Matrices $A_i$ de dimensiones $p_{i-1}\times p_i$:
$$m[i][j]=\min_{i\le k<j}\big(m[i][k]+m[k+1][j]+p_{i-1}p_kp_j\big)$$
$O(n^3)$. Patrón general de DP por intervalos (también: parentizaciones óptimas, árboles BST óptimos).

## 6.8 Viajante exacto: Held-Karp ⭐⭐⭐
Estado: subconjunto visitado $S$ (bitmask) y último nodo $j$.
$$dp[S][j]=\min_{i\in S\setminus\{j\}}\big(dp[S\setminus\{j\}][i]+d_{ij}\big)$$
$O(n^2 2^n)$ en tiempo, $O(n2^n)$ en memoria. Sirve hasta $n\approx20$.

## 6.9 DP sobre dígitos y sobre bitmasks ⭐⭐⭐
Contar números en $[0,N]$ con cierta propiedad: estado (posición, "¿ya menor que N?", estado auxiliar). DP de bitmasks: asignaciones, recubrimientos, hamiltonianos.

## 6.10 Optimizaciones de DP ⭐⭐⭐
*Convex hull trick* (recurrencias $dp[i]=\min_j(m_j x_i+b_j)$), *divide & conquer optimization*, *Knuth optimization* ($O(n^2)\to O(n\log n)$/$O(n^2)$ según caso), "DP con ventana deslizante" (cola monótona).

---

# 7. Cadenas de texto

## 7.1 KMP (Knuth-Morris-Pratt) ⭐⭐
**Idea:** no retroceder en el texto; al fallar, usar la **función de fallo** $\pi[i]$ = longitud del mayor prefijo propio de `P[0..i]` que también es sufijo.
```
pi[0]=0; k=0
for i in 1..m-1:
    while k>0 and P[i]!=P[k]: k=pi[k-1]
    if P[i]==P[k]: k++
    pi[i]=k
# búsqueda: igual, recorriendo T; si k==m → coincidencia en i-m+1; k=pi[k-1]
```
$O(n+m)$.

## 7.2 Rabin-Karp ⭐⭐
Hash polinómico rodante:
$$H(s_{i..i+m-1})=\Big(\sum_{j=0}^{m-1}s_{i+j}\,b^{m-1-j}\Big)\bmod q$$
Deslizar: $H_{i+1}=\big((H_i-s_i b^{m-1})\,b+s_{i+m}\big)\bmod q$. Ante hash igual, verificar. Esperado $O(n+m)$. Útil para múltiples patrones y detección de duplicados. Usa módulo grande (o dos módulos).

## 7.3 Algoritmo Z ⭐⭐
$Z[i]$ = longitud del mayor prefijo común entre `S` y `S[i..]`. Mantener ventana $[l,r)$ más a la derecha:
```
if i<r: Z[i]=min(r-i, Z[i-l]); extender mientras S[Z[i]]==S[i+Z[i]]; actualizar [l,r)
```
Para buscar `P` en `T`: calcular $Z$ sobre `P + '$' + T`; hay coincidencia donde $Z[i]=|P|$. $O(n)$.

## 7.4 Boyer-Moore-Horspool ⭐⭐
Comparar el patrón de derecha a izquierda; al fallar, desplazar según una tabla `shift[c]` = distancia del último `c` (excepto la última posición) al final del patrón. Sublineal en promedio ($\approx n/m$).

## 7.5 Aho-Corasick ⭐⭐⭐
**Problema:** buscar muchos patrones a la vez.
**Idea:** trie de los patrones + **enlaces de fallo** (equivalente a la función $\pi$ de KMP sobre el trie, construidos por BFS) + enlaces "de salida". $O(n+\sum|P_i|+\text{coincidencias})$.

## 7.6 Array de sufijos y LCP ⭐⭐⭐
Ordenar todos los sufijos. **Duplicación de prefijos:** rankear por los primeros $2^k$ caracteres, ordenando pares $(rank[i],rank[i+2^k])$ → $O(n\log^2 n)$ (o $O(n\log n)$ con radix). **Kasai:** LCP en $O(n)$ aprovechando $lcp[rank[i+1]]\ge lcp[rank[i]]-1$. Aplicaciones: subcadena repetida más larga, nº de subcadenas distintas $=\frac{n(n+1)}2-\sum LCP$.

## 7.7 Manacher ⭐⭐⭐
Radios de todos los palíndromos en $O(n)$ reutilizando información del palíndromo más a la derecha (espejo respecto al centro). Trabaja sobre `#a#b#a#` para unificar longitudes pares e impares.

## 7.8 Autómata de sufijos / Hashing de cadenas ⭐⭐⭐
Hash de subcadena en $O(1)$ con prefijos: $H[l..r)=H[r]-H[l]\cdot b^{r-l}\pmod q$. Comparar subcadenas, contar distintas, palíndromos (hash directo vs inverso).

---

# 8. Algoritmos numéricos y teoría de números

## 8.1 Algoritmo de Euclides ⭐
$\gcd(a,b)=\gcd(b,\,a\bmod b)$, $\gcd(a,0)=a$. $O(\log\min(a,b))$ (peor caso: Fibonacci consecutivos, teorema de Lamé). $\mathrm{lcm}(a,b)=a\,b/\gcd(a,b)$.

## 8.2 Euclides extendido ⭐⭐
Encuentra $x,y$ con $ax+by=\gcd(a,b)$ (identidad de Bézout).
```
ext_gcd(a,b): if b==0: return (a,1,0)
    (g,x1,y1)=ext_gcd(b, a%b);  return (g, y1, x1-(a//b)*y1)
```
Inverso modular: si $\gcd(a,m)=1$, $a^{-1}\equiv x\pmod m$ con $ax+my=1$.

## 8.3 Exponenciación rápida ⭐
$a^n$ en $O(\log n)$: $a^n=(a^{n/2})^2$ si $n$ par, $a\cdot a^{n-1}$ si impar. Con módulo: reducir tras cada multiplicación. Funciona con matrices (Fibonacci, recurrencias lineales).
```
pow(a,n,m): r=1; a%=m
  while n>0: if n&1: r=r*a%m; a=a*a%m; n>>=1
```

## 8.4 Criba de Eratóstenes ⭐
Marcar múltiplos de cada primo $p$ empezando en $p^2$, hasta $\sqrt n$. $O(n\log\log n)$. Variante lineal ($O(n)$) guarda el menor factor primo `lp[i]` → factorización rápida.

## 8.5 Test de Miller-Rabin ⭐⭐⭐
**Matemáticas:** si $n$ es primo impar, escribiendo $n-1=2^s d$ con $d$ impar, para todo $a$: $a^d\equiv1$ o $a^{2^r d}\equiv-1\pmod n$ para algún $0\le r<s$. Si falla para un $a$ ⇒ compuesto. Cada base aleatoria deja pasar a un compuesto con prob. $\le1/4$. Determinista en 64 bits con bases $\{2,3,5,7,11,13,17,19,23,29,31,37\}$.

## 8.6 Pollard's rho ⭐⭐⭐
Factorizar enteros grandes: iterar $x_{i+1}=x_i^2+c\bmod n$ y buscar $\gcd(|x_i-x_j|,n)>1$ con la detección de ciclos de Floyd/Brent. Esperado $O(n^{1/4})$ operaciones.

## 8.7 Teorema chino del resto ⭐⭐
Sistema $x\equiv a_i\pmod{m_i}$, $m_i$ coprimos: $x=\sum a_i M_i (M_i^{-1}\bmod m_i)\bmod M$, con $M=\prod m_i$, $M_i=M/m_i$.

## 8.8 Karatsuba ⭐⭐⭐
Multiplicar $x=x_1B+x_0$, $y=y_1B+y_0$ con **3** productos:
$z_2=x_1y_1,\ z_0=x_0y_0,\ z_1=(x_1+x_0)(y_1+y_0)-z_2-z_0$ → $xy=z_2B^2+z_1B+z_0$.
$T(n)=3T(n/2)+O(n)\Rightarrow O(n^{\log_2 3})\approx O(n^{1.585})$.

## 8.9 Strassen ⭐⭐⭐
Multiplica matrices $2\times2$ por bloques con **7** productos en vez de 8 → $O(n^{\log_2 7})\approx O(n^{2.807})$. Solo compensa para $n$ grande; en la práctica se usa BLAS.

## 8.10 Transformada rápida de Fourier (FFT) ⭐⭐⭐
**Matemáticas:** DFT $X_k=\sum_{j=0}^{N-1}x_j\,\omega^{jk}$, $\omega=e^{-2\pi i/N}$. Dividir en pares (E) e impares (O):
$$X_k=E_k+\omega^kO_k,\qquad X_{k+N/2}=E_k-\omega^kO_k$$
$O(N\log N)$ ($N$ potencia de 2). **Multiplicación de polinomios:** FFT de ambos, producto punto a punto, FFT inversa (usar $\omega^{-1}$ y dividir entre $N$). Variante modular: NTT (con $998244353$, raíz primitiva 3).

## 8.11 Bisección y Newton-Raphson ⭐
**Bisección:** si $f(a)f(b)<0$, partir el intervalo; error $\to$ mitad por iteración (convergencia lineal).
**Newton:** $x_{n+1}=x_n-\dfrac{f(x_n)}{f'(x_n)}$. Convergencia cuadrática cerca de la raíz. Raíz cuadrada: $x\leftarrow\frac12(x+a/x)$. Falla si $f'\approx0$ o con mal punto inicial. **Secante:** reemplaza $f'$ por diferencia finita.

## 8.12 Integración numérica ⭐⭐
- Trapecio: $\int_a^b f\approx\frac h2\big[f_0+2f_1+\dots+2f_{n-1}+f_n\big]$, error $O(h^2)$.
- **Simpson** ($n$ par): $\frac h3\big[f_0+4f_1+2f_2+4f_3+\dots+4f_{n-1}+f_n\big]$, error $O(h^4)$.
- **Cuadratura de Gauss-Legendre** (nodos óptimos), **Romberg** (extrapolación de Richardson) y **Monte Carlo** (error $O(1/\sqrt N)$ independiente de la dimensión).

## 8.13 Métodos para EDOs ⭐⭐
Para $y'=f(t,y)$: **Euler** $y_{n+1}=y_n+hf(t_n,y_n)$ (error global $O(h)$). **Runge-Kutta 4:**
$k_1=f(t,y),\ k_2=f(t+\tfrac h2,y+\tfrac h2k_1),\ k_3=f(t+\tfrac h2,y+\tfrac h2k_2),\ k_4=f(t+h,y+hk_3)$
$y_{n+1}=y_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, error $O(h^4)$.

## 8.14 Eliminación gaussiana y LU ⭐⭐
Resolver $Ax=b$ en $O(n^3)$: triangularizar con **pivoteo parcial** (elegir como pivote el mayor $|a_{ik}|$ de la columna, para estabilidad numérica) y sustituir hacia atrás. **LU:** $PA=LU$; resolver $Ly=Pb$, $Ux=y$; sirve para múltiples $b$ y para $\det A=\pm\prod u_{ii}$.

## 8.15 Métodos iterativos lineales ⭐⭐
**Jacobi:** $x_i^{(k+1)}=\frac1{a_{ii}}\big(b_i-\sum_{j\ne i}a_{ij}x_j^{(k)}\big)$. **Gauss-Seidel:** usa ya los valores nuevos. Convergen si $A$ es diagonalmente dominante. **Gradiente conjugado** (SPD): $O(n\sqrt\kappa)$ iteraciones.

## 8.16 Autovalores: iteración de la potencia y QR ⭐⭐⭐
**Potencia:** $v\leftarrow Av/\|Av\|$ converge al autovector dominante; $\lambda\approx v^\top Av$. **Algoritmo QR:** iterar $A_k=Q_kR_k$, $A_{k+1}=R_kQ_k$ → converge a una forma triangular con autovalores en la diagonal. **Jacobi** para simétricas.

## 8.17 Mínimos cuadrados con QR ⭐⭐⭐
$\min\|Ax-b\|$: factorizar $A=QR$ (Gram-Schmidt modificado o Householder) y resolver $Rx=Q^\top b$. Más estable que las ecuaciones normales $A^\top Ax=A^\top b$ (que elevan al cuadrado el número de condición).

---

# 9. Geometría computacional

**Primitiva clave (producto vectorial 2D):** $\text{cross}(o,a,b)=(a_x-o_x)(b_y-o_y)-(a_y-o_y)(b_x-o_x)$. $>0$ giro antihorario, $<0$ horario, $=0$ colineales.

## 9.1 Envolvente convexa: Andrew (monotone chain) ⭐⭐
Ordenar puntos por $(x,y)$; construir cadena inferior y superior descartando el último punto mientras `cross(...) <= 0`.
```
for p in sorted(P): while len(H)>=2 and cross(H[-2],H[-1],p)<=0: H.pop(); H.append(p)
```
$O(n\log n)$. **Graham scan** ordena por ángulo polar. **Jarvis march** (envolver regalo): $O(nh)$, bueno si $h$ (puntos del casco) es pequeño. **Chan:** $O(n\log h)$.

## 9.2 Área de un polígono (fórmula del cordón) ⭐
$A=\frac12\Big|\sum_{i}(x_iy_{i+1}-x_{i+1}y_i)\Big|$. Signo = orientación.

## 9.3 Punto en polígono ⭐⭐
**Ray casting:** lanzar un rayo horizontal y contar cruces con aristas; impar ⇒ dentro. Cuidar vértices y aristas horizontales. **Winding number** para polígonos no simples.

## 9.4 Intersección de segmentos ⭐⭐
$AB$ y $CD$ se cortan si los signos de `cross(A,B,C)` y `cross(A,B,D)` difieren **y** los de `cross(C,D,A)` y `cross(C,D,B)` también; casos colineales: solapamiento de bounding boxes.

## 9.5 Par de puntos más cercano ⭐⭐⭐
Divide y vencerás: ordenar por $x$, dividir, $\delta=\min(\delta_L,\delta_R)$; en la franja $|x-x_m|<\delta$ ordenada por $y$, cada punto solo se compara con los siguientes ~7. $T=2T(n/2)+O(n)=O(n\log n)$.

## 9.6 Barrido de línea (sweep line) ⭐⭐⭐
Procesar eventos ordenados por $x$ manteniendo un conjunto ordenado de objetos "activos". **Shamos-Hoey:** detectar si existe alguna intersección entre $n$ segmentos en $O(n\log n)$. **Bentley-Ottmann:** reportar las $k$ intersecciones en $O((n+k)\log n)$.

## 9.7 Triangulación de Delaunay y Voronoi ⭐⭐⭐
Delaunay maximiza el ángulo mínimo; ningún punto cae dentro de la circunferencia circunscrita de un triángulo (criterio *in-circle*, determinante $4\times4$). Voronoi es su dual. Algoritmos: *flip*, incremental (Bowyer-Watson), $O(n\log n)$.

## 9.8 Estructuras espaciales: k-d tree, Quadtree ⭐⭐
**k-d tree:** partir alternando ejes por la mediana; búsqueda del vecino más cercano en ~$O(\log n)$ con poda por distancia al hiperplano. Se degrada en alta dimensión (maldición de la dimensionalidad). **Quadtree/Octree:** subdivisión espacial para colisiones y rendering.

---

# 10. Compresión, criptografía y códigos

## 10.1 Codificación de Huffman ⭐⭐
**Problema:** código de prefijo óptimo según frecuencias.
```
PQ = hojas (freq,símbolo)
while len(PQ)>1: a=pop_min; b=pop_min; push(Nodo(a.f+b.f, a, b))
# asignar 0/1 a izquierda/derecha; el código de cada hoja = camino
```
$O(k\log k)$. **Matemáticas:** longitud media $L=\sum p_il_i$; límite de Shannon $H\le L<H+1$.

## 10.2 RLE ⭐
Sustituir rachas `aaaabb` por `(a,4)(b,2)`. Eficaz en datos con repeticiones largas (imágenes con áreas planas).

## 10.3 LZ77 / LZ78 / LZW ⭐⭐⭐
**LZ77:** ventana deslizante; emitir triplas `(distancia, longitud, siguiente_carácter)`. Base de DEFLATE (zip, gzip, PNG; LZ77 + Huffman).
**LZW:** diccionario inicial con los 256 bytes; leer la cadena más larga `w` en el diccionario, emitir su código y añadir `w+c`.

## 10.4 Codificación aritmética ⭐⭐⭐
Representa el mensaje como un subintervalo de $[0,1)$ que se va estrechando según $P(\text{símbolo})$; se alcanza la entropía sin la pérdida de "bits enteros" de Huffman. Usa aritmética entera con renormalización.

## 10.5 RSA ⭐⭐
**Matemáticas:** elegir primos $p,q$; $n=pq$; $\varphi(n)=(p-1)(q-1)$; $e$ coprimo con $\varphi(n)$; $d\equiv e^{-1}\pmod{\varphi(n)}$ (Euclides extendido).
Cifrar: $c=m^e\bmod n$. Descifrar: $m=c^d\bmod n$ (válido por el teorema de Euler). Seguridad: factorizar $n$ es difícil. **En producción:** relleno OAEP/PSS, nunca "RSA de libro".

## 10.6 Diffie-Hellman ⭐⭐
Primo $p$, generador $g$. A elige $a$ → envía $A=g^a\bmod p$; B elige $b$ → envía $B=g^b\bmod p$. Secreto común: $B^a=A^b=g^{ab}\bmod p$. Seguridad: problema del logaritmo discreto. Versión en curvas elípticas (ECDH): suma de puntos y multiplicación escalar.

## 10.7 Funciones hash criptográficas (SHA-256) ⭐⭐⭐
Construcción **Merkle-Damgård**: rellenar el mensaje (bit 1, ceros, longitud en 64 bits), procesar bloques de 512 bits con 64 rondas de operaciones `rotr`, `xor`, `and`, suma módulo $2^{32}$ sobre 8 registros de 32 bits. Propiedades: resistencia a preimagen, a segunda preimagen y a colisiones.

## 10.8 Códigos de Hamming (7,4) ⭐⭐
Añadir 3 bits de paridad en posiciones 1,2,4. El **síndrome** $s=Hr^\top\bmod2$ da la posición del bit erróneo (corrige 1 error, detecta 2). Parientes: CRC (división de polinomios en GF(2)), Reed-Solomon (códigos sobre cuerpos finitos).

## 10.9 Cifrados clásicos ⭐
César $c=(m+k)\bmod26$; Vigenère $c_i=(m_i+k_{i\bmod|k|})\bmod26$; rompibles con análisis de frecuencias e índice de coincidencia.

---

# 11. Aleatoriedad y estructuras probabilísticas

## 11.1 Generador congruencial lineal (LCG) ⭐
$X_{n+1}=(aX_n+c)\bmod m$. Periodo completo si (Hull-Dobell): $\gcd(c,m)=1$, $a-1$ divisible por todos los primos de $m$, y por 4 si $4|m$. Rápido pero de baja calidad. Mejores: **xorshift**, **PCG**, **Mersenne Twister** (periodo $2^{19937}-1$).

## 11.2 Muestreo de distribuciones ⭐⭐
- **Transformada inversa:** $X=F^{-1}(U)$, $U\sim\mathcal U(0,1)$. Exponencial: $X=-\ln(U)/\lambda$.
- **Box-Muller (normal):** $z_0=\sqrt{-2\ln u_1}\cos(2\pi u_2)$, $z_1=\sqrt{-2\ln u_1}\sin(2\pi u_2)$.
- **Rechazo:** proponer $x\sim g$, aceptar con prob. $\frac{f(x)}{Mg(x)}$, con $f\le Mg$.
- **Categórica:** sumas prefijas + búsqueda binaria, o *alias method* (muestreo en $O(1)$).

## 11.3 Fisher-Yates (barajar) ⭐
```
for i in n-1 down to 1: j=random(0..i); swap(A[i],A[j])
```
Cada permutación con prob. $1/n!$. Trampa: el "shuffle ingenuo" (`j` en $0..n-1$) está sesgado.

## 11.4 Muestreo de reservorio ⭐⭐
Elegir $k$ elementos uniformes de un flujo de longitud desconocida: guardar los $k$ primeros; para el elemento $i$-ésimo ($i>k$), generar $j\in[1,i]$ y, si $j\le k$, sustituir. Cada elemento queda con prob. $k/i$ final.

## 11.5 Monte Carlo ⭐
Estimar $\pi$: lanzar puntos en $[0,1]^2$, $\pi\approx4\cdot\frac{\#\{x^2+y^2<1\}}{N}$. Error estándar $\propto1/\sqrt N$. Reducción de varianza: muestreo por importancia, variables antitéticas, estratificación.

## 11.6 Filtro de Bloom ⭐⭐
Bits `m` y $k$ hashes. Insertar: activar $k$ posiciones; consultar: todas activas ⇒ "probablemente está", alguna a 0 ⇒ "seguro que no". Falsos positivos:
$$p\approx\Big(1-e^{-kn/m}\Big)^k,\qquad k_{opt}=\frac mn\ln2$$

## 11.7 Count-Min Sketch ⭐⭐
Matriz $d\times w$ con $d$ hashes. Incrementar `T[i][h_i(x)]`; estimar frecuencia como $\min_i T[i][h_i(x)]$. Nunca subestima; error $\le\varepsilon N$ con prob. $1-\delta$ para $w=\lceil e/\varepsilon\rceil$, $d=\lceil\ln(1/\delta)\rceil$.

## 11.8 HyperLogLog ⭐⭐⭐
Contar elementos distintos con $\sim1.5$ KB. Hashear, usar los primeros $b$ bits para elegir registro $j$ (de $m=2^b$) y guardar el máximo de $\rho$ (posición del primer 1 en el resto). Estimación:
$$E=\alpha_m\,m^2\Big/\sum_{j=1}^m2^{-M_j}$$
con corrección para valores pequeños (*linear counting*). Error relativo $\approx1.04/\sqrt m$.

## 11.9 MinHash y LSH ⭐⭐⭐
Similitud de Jaccard $J(A,B)=|A\cap B|/|A\cup B|$. Con permutación aleatoria $\pi$: $P[\min\pi(A)=\min\pi(B)]=J(A,B)$. Firmas de $k$ minhashes → estimación de $J$. **LSH:** dividir la firma en bandas; pares que coinciden en alguna banda son candidatos (curva en S: $1-(1-s^r)^b$).

---

# 12. Backtracking, juegos y búsqueda en IA clásica

## 12.1 Backtracking ⭐⭐
Construir la solución paso a paso y **retroceder** al detectar inconsistencia (poda).
```
solve(estado): if es_solución: guardar; return
    for opción in candidatos(estado):
        if válida: aplicar; solve(estado'); deshacer
```
Ejemplos: **N-Reinas** (columnas, diagonales $r-c$ y $r+c$ con conjuntos o bitmasks), **Sudoku** (heurística MRV: elegir la casilla con menos candidatos), permutaciones, subconjuntos, coloreado de grafos.

## 12.2 Ramificación y poda (Branch & Bound) ⭐⭐⭐
Backtracking para optimización: calcular una **cota** (p. ej. relajación lineal de la mochila) y descartar ramas cuya cota no mejore la mejor solución conocida.

## 12.3 Profundización iterativa (IDDFS) e IDA* ⭐⭐
DFS con límite de profundidad creciente: combina memoria $O(d)$ de DFS con optimalidad de BFS. IDA* limita por $f=g+h$. Típico en puzles (8-puzzle, cubo de Rubik).

## 12.4 Minimax ⭐⭐
Juegos de suma cero de 2 jugadores:
$$V(s)=\begin{cases}u(s)&\text{terminal}\\ \max_a V(s_a)&\text{turno de MAX}\\ \min_a V(s_a)&\text{turno de MIN}\end{cases}$$
$O(b^d)$. Con función de evaluación a profundidad limitada.

## 12.5 Poda alfa-beta ⭐⭐
Mantener $\alpha$ (mejor para MAX hasta ahora) y $\beta$ (mejor para MIN); cortar si $\alpha\ge\beta$. Con ordenación perfecta de movimientos: $O(b^{d/2})$. Añadir tablas de transposición (hash de Zobrist) y profundización iterativa.

## 12.6 Monte Carlo Tree Search (MCTS / UCT) ⭐⭐⭐
Cuatro fases repetidas: **selección** (bajar por el hijo con mayor UCB), **expansión**, **simulación** (rollout aleatorio), **retropropagación**.
$$UCB1=\frac{w_i}{n_i}+c\sqrt{\frac{\ln N}{n_i}},\quad c\approx\sqrt2$$
Base de AlphaGo/AlphaZero (con una red que sustituye al rollout y aporta priors).

## 12.7 Satisfacibilidad: DPLL y CDCL ⭐⭐⭐
**DPLL:** backtracking + *propagación unitaria* + *eliminación de literales puros*. **CDCL:** al encontrar un conflicto, aprende una cláusula, y hace *backjumping*; usa heurística VSIDS y reinicios. Base de los SAT solvers modernos.

## 12.8 Satisfacción de restricciones: AC-3 ⭐⭐⭐
Consistencia de arcos: para cada arco $(X_i,X_j)$ eliminar de $D_i$ los valores sin soporte en $D_j$; reencolar arcos afectados. $O(ed^3)$. Se combina con backtracking (*forward checking*, *MAC*).

---

# 13. Optimización continua

Objetivo: $\min_\theta L(\theta)$. Las condiciones de optimalidad: $\nabla L(\theta^*)=0$ y $\nabla^2L(\theta^*)\succeq0$.

## 13.1 Descenso del gradiente ⭐
$$\theta_{t+1}=\theta_t-\eta\,\nabla L(\theta_t)$$
**Matemáticas:** para $L$ convexa y $\nabla L$ $\beta$-Lipschitz, con $\eta\le1/\beta$ converge como $O(1/t)$; en cuadráticas, la tasa depende del número de condición $\kappa=\lambda_{max}/\lambda_{min}$. **Trampas:** $\eta$ demasiado grande diverge; variables sin normalizar → valle alargado y zigzag. **Comprobación de gradiente:** $\frac{L(\theta+\epsilon e_i)-L(\theta-\epsilon e_i)}{2\epsilon}$ con $\epsilon\approx10^{-5}$.

## 13.2 SGD y minibatch ⭐
Reemplazar $\nabla L$ por el gradiente de un minibatch $B$: $g=\frac1{|B|}\sum_{i\in B}\nabla\ell_i$. Estimador insesgado, ruidoso, y mucho más barato. Barajar los datos en cada época. Para converger sin ruido: tasa decreciente con $\sum\eta_t=\infty$, $\sum\eta_t^2<\infty$.

## 13.3 Momentum y Nesterov ⭐⭐
$v_{t+1}=\beta v_t+\nabla L(\theta_t)$; $\theta_{t+1}=\theta_t-\eta v_{t+1}$ ($\beta\approx0.9$). **Nesterov:** gradiente evaluado en el punto adelantado $\theta_t-\eta\beta v_t$.

## 13.4 AdaGrad, RMSProp ⭐⭐
**AdaGrad:** $G_t=G_{t-1}+g_t^2$; $\theta\leftarrow\theta-\frac{\eta}{\sqrt{G_t}+\epsilon}g_t$ (la tasa de cada parámetro decrece).
**RMSProp:** media móvil $s_t=\rho s_{t-1}+(1-\rho)g_t^2$ ($\rho=0.9$); $\theta\leftarrow\theta-\frac{\eta}{\sqrt{s_t}+\epsilon}g_t$.

## 13.5 Adam / AdamW ⭐⭐
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad\hat v_t=\frac{v_t}{1-\beta_2^t},\quad\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
Valores por defecto: $\eta=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. **AdamW:** el *weight decay* se aplica por separado: $\theta\leftarrow\theta-\eta\lambda\theta$, no a través del gradiente.

## 13.6 Planificadores de tasa de aprendizaje ⭐⭐
*Step decay*, exponencial $\eta_t=\eta_0e^{-kt}$, **coseno** $\eta_t=\eta_{min}+\frac12(\eta_0-\eta_{min})\big(1+\cos\frac{\pi t}{T}\big)$, *warmup* lineal inicial (esencial en Transformers), *cyclical* / *one-cycle*.

## 13.7 Método de Newton ⭐⭐
$\theta_{t+1}=\theta_t-H^{-1}\nabla L$ con $H=\nabla^2L$. Convergencia cuadrática cerca del óptimo; coste $O(d^3)$ por iteración. Ajustar con *line search* o regiones de confianza. En regresión logística: **IRLS**.

## 13.8 BFGS y L-BFGS ⭐⭐⭐
Aproximar $H^{-1}$ iterativamente con actualización de rango 2:
$$H_{k+1}=\Big(I-\rho s y^\top\Big)H_k\Big(I-\rho y s^\top\Big)+\rho ss^\top,\quad s=\theta_{k+1}-\theta_k,\ y=\nabla_{k+1}-\nabla_k,\ \rho=\tfrac1{y^\top s}$$
**L-BFGS** guarda solo los últimos $m$ pares $(s,y)$ → memoria $O(md)$.

## 13.9 Búsqueda de línea (Armijo / Wolfe) ⭐⭐
Elegir el paso $\alpha$ con *backtracking*: reducir $\alpha\leftarrow\tau\alpha$ hasta que $L(\theta+\alpha p)\le L(\theta)+c_1\alpha\nabla L^\top p$ (condición de Armijo, $c_1\approx10^{-4}$).

## 13.10 Gradiente conjugado ⭐⭐⭐
Direcciones $A$-conjugadas ($p_i^\top Ap_j=0$) para $\min\frac12x^\top Ax-b^\top x$: termina en $\le n$ pasos teóricos. $\alpha_k=\frac{r_k^\top r_k}{p_k^\top Ap_k}$, $\beta_k=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}$, $p_{k+1}=r_{k+1}+\beta_kp_k$.

## 13.11 Descenso por coordenadas ⭐⭐
Optimizar una variable cada vez, manteniendo el resto fijas. Muy eficaz en Lasso (cada paso es una soft-threshold cerrada).

## 13.12 Programación lineal: Símplex ⭐⭐⭐
$\max c^\top x$ s.a. $Ax\le b,\ x\ge0$. El óptimo está en un vértice del politopo. Añadir variables de holgura, construir el *tableau* y pivotar: elegir columna con coeficiente reducido positivo (entrante), fila por el **test del cociente mínimo** (saliente). Peor caso exponencial, excelente en la práctica. Alternativas: puntos interiores (polinomial). **Dualidad:** $\max c^\top x=\min b^\top y$ s.a. $A^\top y\ge c$.

## 13.13 Multiplicadores de Lagrange y KKT ⭐⭐⭐
Con restricciones de igualdad: $\mathcal L(x,\lambda)=f(x)+\lambda^\top g(x)$, $\nabla_x\mathcal L=0$, $g=0$. Con desigualdades $h(x)\le0$: condiciones **KKT** (factibilidad, $\mu\ge0$, holgura complementaria $\mu_ih_i=0$). Base de las SVM.

## 13.14 ADMM y gradiente proximal ⭐⭐⭐
**Proximal (ISTA):** $x\leftarrow\mathrm{prox}_{\eta g}\big(x-\eta\nabla f(x)\big)$; para $g=\lambda\|\cdot\|_1$, $\mathrm{prox}$ = *soft-threshold* $S_\tau(z)=\mathrm{sign}(z)\max(|z|-\tau,0)$.

---

# 14. Metaheurísticas y algoritmos evolutivos

Para problemas sin gradiente, discretos o con muchos óptimos locales. **No garantizan el óptimo.**

## 14.1 Escalada (Hill Climbing) ⭐
Moverse al mejor vecino mientras mejore. Variantes: *steepest ascent*, estocástica, **reinicios aleatorios**. Se atasca en óptimos locales.

## 14.2 Recocido simulado (Simulated Annealing) ⭐⭐
**Idea:** aceptar empeoramientos con probabilidad decreciente (analogía con el enfriamiento de metales).
**Matemáticas:** si $\Delta=f(x')-f(x)>0$ (minimización), aceptar con prob. $e^{-\Delta/T}$ (criterio de Metropolis). Enfriamiento geométrico $T_{k+1}=\alpha T_k$ ($\alpha\in[0.90,0.99]$). Con enfriamiento logarítmico $T_k=c/\ln(1+k)$ converge al óptimo global (en teoría).
```
x=x0; T=T0
while T>Tmin: x'=vecino(x); d=f(x')-f(x)
    if d<0 or rand()<exp(-d/T): x=x'
    T*=alpha
```

## 14.3 Búsqueda tabú ⭐⭐
Escalada permitiendo movimientos que empeoren, con una **lista tabú** (memoria a corto plazo) que prohíbe deshacer movimientos recientes durante $k$ iteraciones; **criterio de aspiración**: se permite un movimiento tabú si mejora el mejor global.

## 14.4 Algoritmo genético (GA) ⭐⭐
**Idea:** población de cromosomas que evoluciona por selección, cruce y mutación.
```
P = población aleatoria
repeat: evaluar fitness
    seleccionar padres (torneo / ruleta / ranking)
    hijos = cruce(padres) ; mutar(hijos) ; elitismo (conservar los mejores)
    P = nueva generación
```
- **Selección ruleta:** $p_i=\frac{f_i}{\sum f_j}$. **Torneo:** el mejor de $k$ individuos al azar.
- **Cruce:** un punto, dos puntos, uniforme; **OX / PMX** para permutaciones (TSP); **SBX** para reales.
- **Mutación:** inversión de bit con prob. $\approx1/L$; gaussiana $x+\mathcal N(0,\sigma^2)$ para reales; *swap* / inversión para permutaciones.
- **Teorema de esquemas (Holland):** $E[m(H,t+1)]\ge m(H,t)\frac{f(H)}{\bar f}\Big(1-p_c\frac{\delta(H)}{L-1}-o(H)p_m\Big)$.

## 14.5 Optimización por enjambre de partículas (PSO) ⭐⭐
Cada partícula tiene posición $x_i$, velocidad $v_i$, su mejor posición $p_i$ y el mejor del enjambre $g$:
$$v_i\leftarrow w\,v_i+c_1r_1(p_i-x_i)+c_2r_2(g-x_i),\qquad x_i\leftarrow x_i+v_i$$
con $r_1,r_2\sim\mathcal U(0,1)$, $w\approx0.7$, $c_1=c_2\approx1.5$. Limitar $|v|$.

## 14.6 Colonia de hormigas (ACO) ⭐⭐⭐
Para caminos (TSP). La hormiga $k$ en la ciudad $i$ elige $j$ con probabilidad
$$p_{ij}=\frac{\tau_{ij}^\alpha\,\eta_{ij}^\beta}{\sum_{l\in\text{permitidos}}\tau_{il}^\alpha\eta_{il}^\beta},\qquad\eta_{ij}=\frac1{d_{ij}}$$
Evaporación y depósito: $\tau_{ij}\leftarrow(1-\rho)\tau_{ij}+\sum_k\Delta\tau_{ij}^k$, con $\Delta\tau^k_{ij}=Q/L_k$ si la hormiga $k$ usó la arista.

## 14.7 Evolución diferencial (DE) ⭐⭐
Por cada individuo $x_i$: mutante $v=x_{r_1}+F(x_{r_2}-x_{r_3})$ ($F\in[0.4,1]$); cruce binomial con prob. $CR$ para obtener el *trial* $u$; sustituir $x_i$ por $u$ si $f(u)\le f(x_i)$. Excelente optimizador global para variables continuas con pocos parámetros.

## 14.8 Estrategias evolutivas y CMA-ES ⭐⭐⭐
**$(\mu,\lambda)$-ES:** generar $\lambda$ hijos $x=m+\sigma\mathcal N(0,C)$, seleccionar los $\mu$ mejores. **CMA-ES** adapta la media $m$, el paso $\sigma$ y la matriz de covarianza $C$ usando los mejores individuos (aprende la geometría del paisaje). Estado del arte en optimización continua de caja negra.

## 14.9 NSGA-II (multiobjetivo) ⭐⭐⭐
Dominancia de Pareto: $a\prec b$ si $a$ no es peor en ningún objetivo y es mejor en alguno. Ordenar en **frentes no dominados** y desempatar con la **distancia de apiñamiento** (*crowding distance*) para preservar diversidad. Resultado: aproximación del frente de Pareto.

## 14.10 Programación genética y NEAT ⭐⭐⭐
**GP:** individuos = árboles de expresión; cruce = intercambiar subárboles. **NEAT:** evoluciona topología y pesos de redes neuronales, protegiendo innovaciones con especiación y *historical markings*.

## 14.11 Optimización bayesiana ⭐⭐⭐
Para funciones carísimas de evaluar (hiperparámetros): ajustar un **proceso gaussiano** como modelo sustituto y elegir el siguiente punto maximizando una *función de adquisición*, p. ej. **Mejora Esperada**:
$$EI(x)=(\mu-f^*)\Phi(z)+\sigma\varphi(z),\quad z=\frac{\mu-f^*}{\sigma}$$
($\Phi,\varphi$: CDF y PDF normales estándar; maximización con $f^*$ el mejor valor observado.)

---

# 15. Machine Learning supervisado

**Marco general:** datos $(x_i,y_i)$, modelo $f_\theta$, pérdida $\ell$. Minimizar el **riesgo empírico** $\frac1n\sum\ell(f_\theta(x_i),y_i)+\lambda\,\Omega(\theta)$. Sesgo-varianza: $E[(y-\hat f)^2]=\text{sesgo}^2+\text{varianza}+\sigma^2_{ruido}$.

## 15.1 Regresión lineal (OLS) ⭐
**Modelo:** $\hat y=X\beta$ con $X$ incluyendo una columna de unos.
**Pérdida:** $L=\frac1n\|X\beta-y\|^2$.
**Solución cerrada** (ecuaciones normales): $\beta=(X^\top X)^{-1}X^\top y$. **Gradiente:** $\nabla L=\frac2nX^\top(X\beta-y)$.
**Trampas:** normalizar variables antes de usar GD; $X^\top X$ singular con variables colineales (usar pseudoinversa o regularización). **Métricas:** $R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$, RMSE.

## 15.2 Regresión polinómica ⭐
Expandir características $[1,x,x^2,\dots,x^d]$ y aplicar OLS. Cuanto mayor $d$, mayor riesgo de sobreajuste.

## 15.3 Ridge (L2) ⭐⭐
$L=\|X\beta-y\|^2+\lambda\|\beta\|^2$ → $\beta=(X^\top X+\lambda I)^{-1}X^\top y$. Siempre invertible. Interpretación bayesiana: prior gaussiano sobre $\beta$. No regularizar el término independiente.

## 15.4 Lasso (L1) y Elastic Net ⭐⭐
$L=\frac1{2n}\|X\beta-y\|^2+\lambda\|\beta\|_1$. Produce **soluciones dispersas**. Sin forma cerrada; por descenso por coordenadas: $\beta_j\leftarrow\dfrac{S_{\lambda}(\rho_j)}{z_j}$, con $\rho_j=\frac1n X_j^\top(y-X_{-j}\beta_{-j})$, $z_j=\frac1n\|X_j\|^2$. **Elastic Net:** mezcla $\lambda_1\|\beta\|_1+\lambda_2\|\beta\|^2$.

## 15.5 Regresión logística ⭐⭐
**Modelo:** $p=\sigma(w^\top x+b)$, $\sigma(z)=\frac1{1+e^{-z}}$, $\sigma'=\sigma(1-\sigma)$.
**Pérdida (entropía cruzada binaria):** $L=-\frac1n\sum\big[y\log p+(1-y)\log(1-p)\big]$.
**Gradiente:** $\nabla_wL=\frac1nX^\top(p-y)$, $\nabla_bL=\frac1n\sum(p_i-y_i)$. ¡Misma forma que OLS!
**Multiclase (softmax):** $p_k=\frac{e^{z_k}}{\sum_je^{z_j}}$, gradiente $\frac1nX^\top(P-Y)$ con $Y$ one-hot. **Estabilidad numérica:** restar $\max z$ antes de la exponencial.

## 15.6 k-Vecinos más cercanos (k-NN) ⭐
Predecir por voto mayoritario (clasificación) o media (regresión) de los $k$ vecinos más próximos. Distancias: euclídea, Manhattan, coseno $1-\frac{x\cdot y}{\|x\|\|y\|}$. **Normalizar variables.** Con k-d tree o HNSW/FAISS para acelerar. Sin entrenamiento, predicción $O(nd)$.

## 15.7 Naive Bayes ⭐⭐
Asumir independencia condicional: $P(y|x)\propto P(y)\prod_jP(x_j|y)$. Trabajar en logaritmos: $\hat y=\arg\max\big[\log P(y)+\sum_j\log P(x_j|y)\big]$.
- **Gaussiano:** $P(x_j|y)=\mathcal N(x_j|\mu_{jy},\sigma^2_{jy})$.
- **Multinomial** (texto): $P(w|y)=\frac{N_{wy}+\alpha}{N_y+\alpha|V|}$ (suavizado de Laplace).

## 15.8 Árboles de decisión (CART / ID3 / C4.5) ⭐⭐
**Idea:** partir recursivamente el espacio con umbrales $x_j\le t$ que maximicen la ganancia.
**Impureza:** Gini $G=1-\sum_kp_k^2$; entropía $H=-\sum p_k\log_2p_k$.
**Ganancia:** $\Delta=I(\text{padre})-\sum_{c}\frac{n_c}{n}I(c)$. Regresión: reducción de varianza (MSE).
Probar como umbrales los puntos medios entre valores consecutivos ordenados. Parar por profundidad, mínimo de muestras o **poda** por coste-complejidad: $R_\alpha(T)=R(T)+\alpha|T|$. Se sobreajustan fácilmente.

## 15.9 Bagging y Random Forest ⭐⭐
Entrenar $B$ árboles sobre muestras *bootstrap* (con reemplazo) y, en cada partición, considerar solo $m\approx\sqrt d$ características al azar. Predicción: voto/media. Reduce varianza. **Error OOB:** evaluar cada muestra con los árboles que no la vieron. Importancia de variables por permutación.

## 15.10 AdaBoost ⭐⭐⭐
Pesos de muestras $w_i=1/n$. Para cada ronda $m$: ajustar clasificador débil $h_m$ con error ponderado $\varepsilon_m$; $\alpha_m=\frac12\ln\frac{1-\varepsilon_m}{\varepsilon_m}$; actualizar $w_i\leftarrow w_i\,e^{-\alpha_my_ih_m(x_i)}$ y normalizar ($y\in\{-1,+1\}$). Modelo final: $\mathrm{sign}\big(\sum_m\alpha_mh_m(x)\big)$.

## 15.11 Gradient Boosting (GBM, XGBoost, LightGBM) ⭐⭐⭐
**Idea:** modelo aditivo $F_m=F_{m-1}+\nu\,h_m$ donde cada árbol $h_m$ se ajusta al **gradiente negativo de la pérdida** (pseudo-residuos): $r_i=-\frac{\partial\ell(y_i,F(x_i))}{\partial F(x_i)}\Big|_{F_{m-1}}$. Para MSE son los residuos $y-F$; para logística, $y-p$.
**XGBoost** usa expansión de segundo orden con $g_i,h_i$ (gradiente y hessiano): peso óptimo de hoja $w^*=-\frac{\sum g}{\sum h+\lambda}$ y ganancia de partición
$$\text{Gain}=\frac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma$$
**LightGBM:** crecimiento por hojas (*leaf-wise*) y histogramas de características.

## 15.12 Máquinas de vectores de soporte (SVM) ⭐⭐⭐
**Margen duro:** $\min\frac12\|w\|^2$ s.a. $y_i(w^\top x_i+b)\ge1$. **Margen blando:** $\min\frac12\|w\|^2+C\sum\xi_i$ ≡ **hinge loss** $\max(0,1-y_i f(x_i))$.
**Dual:** $\max_\alpha\sum\alpha_i-\frac12\sum_{ij}\alpha_i\alpha_jy_iy_jK(x_i,x_j)$ s.a. $0\le\alpha_i\le C$, $\sum\alpha_iy_i=0$. Predicción: $f(x)=\sum\alpha_iy_iK(x_i,x)+b$.
**Núcleos:** lineal $x^\top z$, polinómico $(\gamma x^\top z+r)^d$, **RBF** $e^{-\gamma\|x-z\|^2}$. Resolver con **SMO** (optimizar pares $(\alpha_i,\alpha_j)$ analíticamente) o SGD sobre hinge en el caso lineal.

## 15.13 Análisis discriminante lineal (LDA) ⭐⭐
Gaussianas con la misma covarianza $\Sigma$: frontera lineal $\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k$. Proyección de **Fisher:** $w\propto S_W^{-1}(\mu_1-\mu_2)$.

## 15.14 Regresión de procesos gaussianos ⭐⭐⭐
Predicción en $x_*$: $\mu_*=k_*^\top(K+\sigma_n^2I)^{-1}y$, $\sigma_*^2=k(x_*,x_*)-k_*^\top(K+\sigma_n^2I)^{-1}k_*$. Coste $O(n^3)$ (Cholesky). Núcleo RBF: $k(x,x')=\sigma_f^2e^{-\|x-x'\|^2/2\ell^2}$.

## 15.15 Evaluación y selección de modelos ⭐⭐
- **Validación cruzada k-fold**, estratificada para clasificación; *nested CV* para ajustar hiperparámetros. **Nunca** ajustar escaladores sobre todos los datos (fuga de información).
- **Matriz de confusión:** Precisión $=\frac{TP}{TP+FP}$, Recall $=\frac{TP}{TP+FN}$, $F_1=\frac{2PR}{P+R}$, Exactitud.
- **ROC-AUC:** área bajo TPR vs FPR; equivale a $P(\text{score}_+>\text{score}_-)$. **PR-AUC** si hay desbalanceo.
- **Calibración:** Brier score, *Platt scaling*, regresión isotónica.

---

# 16. Machine Learning no supervisado

## 16.1 K-means (algoritmo de Lloyd) ⭐
**Objetivo:** $J=\sum_{k}\sum_{x\in C_k}\|x-\mu_k\|^2$.
```
inicializar centroides μ_1..μ_K
repeat: asignar cada x al centroide más cercano
        μ_k = media de los puntos asignados a k
until no cambian
```
Cada paso reduce $J$ → converge a un mínimo **local**. $O(nKdI)$. **Trampas:** depende de la inicialización (reiniciar varias veces, quedarse con menor $J$); clusters vacíos; asume clusters esféricos de tamaño similar. **Elegir K:** método del codo, *silhouette* $s=\frac{b-a}{\max(a,b)}$, gap statistic.

## 16.2 K-means++ ⭐⭐
Inicialización: primer centro al azar; cada nuevo centro $x$ con probabilidad $\propto D(x)^2$, la distancia al centro más cercano ya elegido. Garantiza $O(\log K)$-aproximación esperada.

## 16.3 K-medoids (PAM) ⭐⭐
Como K-means, pero los centros son puntos reales y minimiza distancias arbitrarias (no solo euclídea). Más robusto frente a *outliers*, $O(K(n-K)^2)$ por iteración.

## 16.4 DBSCAN ⭐⭐
Parámetros $\varepsilon$ y `minPts`. *Punto núcleo*: tiene $\ge$ `minPts` vecinos a distancia $\le\varepsilon$. Expandir clusters conectando puntos alcanzables por densidad; los no alcanzables son ruido. Encuentra formas arbitrarias y detecta *outliers*; no necesita $K$. $O(n\log n)$ con índice espacial. **$\varepsilon$:** gráfico de distancia al $k$-ésimo vecino ordenada (codo). **HDBSCAN** elimina el parámetro $\varepsilon$.

## 16.5 Clustering jerárquico aglomerativo ⭐⭐
Empezar con cada punto como cluster; fusionar repetidamente el par más cercano. Distancia entre clusters (*linkage*): simple (mín), completo (máx), promedio, **Ward** (minimiza el aumento de varianza intra-cluster). Resultado: **dendrograma**. $O(n^2\log n)$.

## 16.6 Mezcla de gaussianas y EM ⭐⭐⭐
**Modelo:** $p(x)=\sum_k\pi_k\,\mathcal N(x|\mu_k,\Sigma_k)$.
**Paso E** (responsabilidades): $\gamma_{ik}=\dfrac{\pi_k\mathcal N(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i|\mu_j,\Sigma_j)}$.
**Paso M:** con $N_k=\sum_i\gamma_{ik}$:
$$\mu_k=\frac1{N_k}\sum_i\gamma_{ik}x_i,\quad\Sigma_k=\frac1{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top,\quad\pi_k=\frac{N_k}n$$
Cada iteración aumenta la log-verosimilitud $\sum_i\log p(x_i)$. **Trampas:** singularidades ($\Sigma\to0$ cuando un cluster colapsa en un punto → añadir $\epsilon I$); implementar con *log-sum-exp*. K-means = caso límite con covarianzas $\sigma^2I\to0$. Elegir $K$ con BIC/AIC.

## 16.7 PCA ⭐⭐
**Idea:** proyectar sobre las direcciones de máxima varianza.
**Matemáticas:** centrar $X$; covarianza $C=\frac1{n-1}X^\top X$; autovectores $v_i$ con autovalores $\lambda_1\ge\lambda_2\ge\dots$. Proyección: $Z=XV_k$. Varianza explicada: $\frac{\sum_{i\le k}\lambda_i}{\sum\lambda_i}$. **Vía SVD:** $X=U\Sigma V^\top$ → $V$ = componentes, $\lambda_i=\sigma_i^2/(n-1)$ (más estable). Estandarizar variables si tienen escalas distintas.

## 16.8 SVD y aproximación de bajo rango ⭐⭐⭐
$A=U\Sigma V^\top$. **Eckart-Young:** la mejor aproximación de rango $k$ (en norma de Frobenius y espectral) es $A_k=\sum_{i\le k}\sigma_iu_iv_i^\top$. Aplicaciones: compresión, LSA (texto), recomendadores, pseudoinversa $A^+=V\Sigma^+U^\top$.

## 16.9 t-SNE ⭐⭐⭐
Visualización 2D/3D que conserva vecindades. Afinidades en alta dimensión $p_{j|i}=\dfrac{e^{-\|x_i-x_j\|^2/2\sigma_i^2}}{\sum_{k\ne i}e^{-\|x_i-x_k\|^2/2\sigma_i^2}}$ ($\sigma_i$ ajustado por **perplejidad**); en baja dimensión usa una $t$ de Student: $q_{ij}=\dfrac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}}$. Minimizar $KL(P\|Q)$ con gradiente $\frac{\partial C}{\partial y_i}=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+\|y_i-y_j\|^2)^{-1}$. **UMAP** es similar pero más rápido y mejor preservando estructura global.

## 16.10 Reglas de asociación: Apriori / FP-Growth ⭐⭐
$\text{soporte}(A)=\frac{\#\{A\subseteq T\}}{N}$, $\text{confianza}(A\Rightarrow B)=\frac{\text{sop}(A\cup B)}{\text{sop}(A)}$, $\text{lift}=\frac{\text{conf}}{\text{sop}(B)}$. **Apriori:** generar candidatos de tamaño $k$ uniendo frecuentes de $k-1$; poda por la propiedad antimonótona (todo subconjunto de un frecuente es frecuente). **FP-Growth:** árbol de prefijos de transacciones, sin generar candidatos.

## 16.11 Isolation Forest (detección de anomalías) ⭐⭐
Árboles con particiones **aleatorias**: las anomalías se aíslan con pocos cortes. Puntuación $s(x,n)=2^{-E[h(x)]/c(n)}$ con $c(n)=2H(n-1)-\frac{2(n-1)}n$ ($H$ = número armónico). $s\to1$ anomalía; $s\ll0.5$ normal. Otros: LOF (densidad local), One-Class SVM, distancia de Mahalanobis $\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$.

## 16.12 Factorización no negativa (NMF) ⭐⭐⭐
$V\approx WH$, $W,H\ge0$. Actualizaciones multiplicativas (Lee-Seung, norma de Frobenius): $H\leftarrow H\odot\frac{W^\top V}{W^\top WH}$, $W\leftarrow W\odot\frac{VH^\top}{WHH^\top}$. Produce partes aditivas interpretables (temas, componentes).

## 16.13 Clustering espectral ⭐⭐⭐
Construir grafo de afinidad $W$, grado $D$, Laplaciano $L=D-W$ (o normalizado $I-D^{-1/2}WD^{-1/2}$); tomar los $k$ autovectores de menor autovalor; aplicar K-means sobre sus filas. Funciona con clusters no convexos.

## 16.14 Sistemas de recomendación: filtrado colaborativo y factorización matricial ⭐⭐
Ratings $R\approx PQ^\top$; minimizar $\sum_{(u,i)\in\Omega}(r_{ui}-p_u^\top q_i)^2+\lambda(\|p_u\|^2+\|q_i\|^2)$ por SGD o **ALS** (alternando resolver mínimos cuadrados para $P$ y $Q$). Basado en vecinos: similitud coseno / Pearson entre usuarios o ítems.

---

# 17. Modelos probabilísticos y secuenciales

## 17.1 Cadenas de Markov ⭐⭐
$P(X_{t+1}|X_t,\dots)=P(X_{t+1}|X_t)$. Matriz de transición $P$ (filas suman 1). Distribución estacionaria: $\pi P=\pi$ (autovector izquierdo con $\lambda=1$), por iteración de la potencia. Ergódica ⇒ converge a $\pi$ desde cualquier inicio.

## 17.2 Modelos ocultos de Markov (HMM): Forward, Viterbi, Baum-Welch ⭐⭐⭐
Parámetros: transiciones $A_{ij}$, emisiones $B_j(o)$, inicial $\pi_i$.
- **Forward:** $\alpha_t(j)=\Big[\sum_i\alpha_{t-1}(i)A_{ij}\Big]B_j(o_t)$ → $P(O)=\sum_j\alpha_T(j)$.
- **Viterbi** (secuencia más probable): $\delta_t(j)=\max_i\big[\delta_{t-1}(i)A_{ij}\big]B_j(o_t)$ guardando `argmax` para retroceder. $O(TN^2)$. **Trabajar en logaritmos** (o escalar) para evitar *underflow*.
- **Baum-Welch (EM):** forward-backward para $\gamma_t(i)$ y $\xi_t(i,j)$; reestimar $A_{ij}=\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(i)}$, $B_j(k)=\frac{\sum_{t:o_t=k}\gamma_t(j)}{\sum_t\gamma_t(j)}$.

## 17.3 Filtro de Kalman ⭐⭐⭐
Sistema lineal gaussiano: $x_t=Fx_{t-1}+w$, $w\sim\mathcal N(0,Q)$; $z_t=Hx_t+v$, $v\sim\mathcal N(0,R)$.
**Predicción:** $\hat x^-=F\hat x$, $P^-=FPF^\top+Q$.
**Actualización:** $K=P^-H^\top(HP^-H^\top+R)^{-1}$, $\hat x=\hat x^-+K(z-H\hat x^-)$, $P=(I-KH)P^-$.
**EKF:** linealiza con jacobianos para modelos no lineales. **UKF:** puntos sigma.

## 17.4 Filtro de partículas ⭐⭐⭐
Aproximar el posterior con $N$ muestras ponderadas: propagar con el modelo de transición, ponderar $w_i\propto p(z|x_i)$, normalizar y **remuestrear** cuando el tamaño efectivo $N_{eff}=1/\sum w_i^2$ cae bajo un umbral.

## 17.5 Metropolis-Hastings (MCMC) ⭐⭐⭐
Muestrear de $p(x)\propto\tilde p(x)$ (sin la constante de normalización). Proponer $x'\sim q(x'|x)$; aceptar con
$$\alpha=\min\Big(1,\frac{\tilde p(x')\,q(x|x')}{\tilde p(x)\,q(x'|x)}\Big)$$
Cumple *balance detallado* ⇒ $p$ es estacionaria. Descartar *burn-in*; tasa de aceptación óptima ≈ 23% (alta dimensión) con propuesta gaussiana. **Gibbs:** actualizar cada variable desde su condicional $p(x_i|x_{-i})$ (siempre se acepta). **HMC:** usa gradientes y dinámica hamiltoniana (base de Stan).

## 17.6 Inferencia bayesiana conjugada ⭐⭐
Beta-Binomial: prior $\mathrm{Beta}(a,b)$, con $k$ éxitos en $n$ → posterior $\mathrm{Beta}(a+k,\,b+n-k)$. Normal-Normal, Dirichlet-Multinomial, Gamma-Poisson. Estimadores: MLE $\arg\max p(D|\theta)$, MAP $\arg\max p(D|\theta)p(\theta)$.

## 17.7 Algoritmo EM general ⭐⭐⭐
Para variables latentes $z$: **E:** $Q(\theta|\theta^{old})=E_{z|x,\theta^{old}}[\log p(x,z|\theta)]$; **M:** $\theta^{new}=\arg\max_\theta Q$. Garantiza $\log p(x|\theta)$ no decreciente (maximiza la cota inferior ELBO).

## 17.8 Inferencia variacional ⭐⭐⭐
Aproximar $p(z|x)$ por $q_\phi(z)$ minimizando $KL(q\|p)$ ≡ maximizar el **ELBO**:
$$\log p(x)\ge E_q[\log p(x,z)]-E_q[\log q(z)]$$
Base de los VAE (cap. 19).

## 17.9 Redes bayesianas y suma-producto ⭐⭐⭐
Grafo dirigido acíclico, $P(X_1..X_n)=\prod_iP(X_i|\text{pa}(X_i))$. **Eliminación de variables** y **propagación de creencias** (paso de mensajes) para marginales exactos en árboles; *junction tree* en general; MCMC/variacional en el resto.

## 17.10 Campos aleatorios condicionales (CRF) ⭐⭐⭐
Modelo discriminativo para secuencias: $P(y|x)=\frac1{Z(x)}\exp\Big(\sum_t\sum_kw_kf_k(y_{t-1},y_t,x,t)\Big)$. Entrenamiento por gradiente de la log-verosimilitud con forward-backward; decodificación con Viterbi (NER, POS tagging).

---

# 18. Aprendizaje por refuerzo

**Marco (MDP):** estados $S$, acciones $A$, transición $P(s'|s,a)$, recompensa $r$, descuento $\gamma\in[0,1)$. Retorno $G_t=\sum_{k\ge0}\gamma^kr_{t+k+1}$.
**Funciones de valor:** $V^\pi(s)=E_\pi[G_t|s]$, $Q^\pi(s,a)=E_\pi[G_t|s,a]$.
**Ecuación de Bellman (óptima):** $Q^*(s,a)=\sum_{s'}P(s'|s,a)\big[r+\gamma\max_{a'}Q^*(s',a')\big]$.

## 18.1 Bandidos multibrazo ⭐⭐
Dilema exploración-explotación. $Q_n=Q_{n-1}+\frac1n(r-Q_{n-1})$ (media incremental).
- **$\varepsilon$-greedy:** con prob. $\varepsilon$ acción aleatoria; si no, la de mayor $Q$.
- **UCB1:** $a=\arg\max\Big[Q_a+c\sqrt{\frac{\ln t}{N_a}}\Big]$.
- **Thompson sampling:** muestrear $\theta_a\sim$ posterior (Beta para Bernoulli) y actuar con $\arg\max\theta_a$.

## 18.2 Iteración de valores y de políticas ⭐⭐
Requiere el modelo del entorno.
**Value iteration:** $V(s)\leftarrow\max_a\sum_{s'}P(s'|s,a)[r+\gamma V(s')]$ hasta $\max|\Delta V|<\theta$ (contracción: $\gamma$). Política: $\pi(s)=\arg\max_a(\dots)$.
**Policy iteration:** alternar evaluación de $\pi$ (resolver Bellman lineal) y mejora greedy.

## 18.3 Monte Carlo y TD(0) ⭐⭐
**MC:** promediar retornos completos de episodios. **TD(0):** $V(s)\leftarrow V(s)+\alpha\big[r+\gamma V(s')-V(s)\big]$; el término entre corchetes es el **error TD** $\delta$. **TD($\lambda$)/GAE:** mezcla geométrica de retornos de $n$ pasos con trazas de elegibilidad.

## 18.4 SARSA ⭐⭐
*On-policy*: $Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma Q(s',a')-Q(s,a)\big]$ con $a'$ la acción **realmente** tomada ($\varepsilon$-greedy).

## 18.5 Q-learning ⭐⭐
*Off-policy*: $Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]$.
```
for episodio: s=reset()
  while not done: a = ε-greedy(Q[s]); (s',r,done)=env.step(a)
      target = r + (0 if done else γ*max(Q[s']))
      Q[s,a] += α*(target - Q[s,a]); s=s'
```
Converge si todo par $(s,a)$ se visita infinitas veces y $\alpha$ decae adecuadamente.

## 18.6 Deep Q-Network (DQN) ⭐⭐⭐
$Q_\theta(s,a)$ con una red neuronal. Dos trucos para estabilizar: **replay buffer** (muestrear transiciones aleatorias, rompe correlaciones) y **red objetivo** $\theta^-$ copiada cada $C$ pasos.
$$L=E\Big[\big(r+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a)\big)^2\Big]$$
Mejoras: **Double DQN** (elegir $a'$ con $\theta$, evaluar con $\theta^-$), *dueling*, replay priorizado, Rainbow.

## 18.7 Gradiente de política: REINFORCE ⭐⭐⭐
**Teorema del gradiente de política:** $\nabla_\theta J=E_\pi\big[\nabla_\theta\log\pi_\theta(a|s)\,G_t\big]$.
Actualizar $\theta\leftarrow\theta+\alpha\,\nabla_\theta\log\pi_\theta(a_t|s_t)\,(G_t-b(s_t))$. Restar una **línea base** $b$ reduce varianza sin sesgo. Política softmax discreta o gaussiana continua.

## 18.8 Actor-Crítico (A2C/A3C) ⭐⭐⭐
Actor $\pi_\theta$ y crítico $V_w$; la **ventaja** $A_t=r_t+\gamma V(s_{t+1})-V(s_t)$ sustituye a $G_t$: $\nabla J=E[\nabla\log\pi\cdot A]$. Pérdida del crítico: $(r+\gamma V(s')-V(s))^2$; añadir bonus de entropía.

## 18.9 PPO ⭐⭐⭐
Razón de probabilidades $\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$.
$$L^{CLIP}=E\Big[\min\big(\rho_tA_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)\Big],\ \epsilon\approx0.2$$
Varias épocas de minibatches sobre el mismo lote, con GAE para $A_t$. Algoritmo por defecto en muchas aplicaciones (RLHF incluido). Predecesor: **TRPO** (región de confianza con KL).

## 18.10 DDPG / SAC / TD3 (acciones continuas) ⭐⭐⭐
Actor determinista $\mu_\theta(s)$ + crítico $Q(s,a)$, con red objetivo y *soft update* $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$. **TD3:** dos críticos (tomar el mínimo), actualización retrasada del actor, ruido en el objetivo. **SAC:** maximiza $E[\sum r+\alpha H(\pi)]$ (entropía máxima), política estocástica.

## 18.11 AlphaZero ⭐⭐⭐
Red con salidas política $p$ y valor $v$; MCTS guiado con selección $a=\arg\max\big[Q+c\,p(a)\frac{\sqrt{N}}{1+N_a}\big]$. Entrenar por *self-play*: pérdida $(z-v)^2-\pi^\top\log p+c\|\theta\|^2$, donde $\pi$ = distribución de visitas de MCTS y $z$ = resultado de la partida.

## 18.12 RLHF y DPO ⭐⭐⭐
**RLHF:** modelo de recompensa entrenado con preferencias (Bradley-Terry: $P(y_w\succ y_l)=\sigma(r(y_w)-r(y_l))$) + PPO con penalización KL a la política de referencia. **DPO:** optimiza directamente $-\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\Big)$ sin modelo de recompensa explícito.

---

# 19. Deep Learning

## 19.1 Perceptrón ⭐
$\hat y=\mathbb 1[w^\top x+b>0]$. Regla: si se equivoca, $w\leftarrow w+\eta(y-\hat y)x$. **Teorema de convergencia:** si los datos son linealmente separables con margen $\gamma$ y $\|x\|\le R$, converge en $\le(R/\gamma)^2$ errores. No resuelve XOR (necesita capa oculta).

## 19.2 Funciones de activación ⭐
| Nombre | $\sigma(z)$ | $\sigma'(z)$ |
|---|---|---|
| Sigmoide | $\frac1{1+e^{-z}}$ | $\sigma(1-\sigma)$ |
| Tanh | $\tanh z$ | $1-\tanh^2z$ |
| ReLU | $\max(0,z)$ | $\mathbb 1[z>0]$ |
| Leaky ReLU | $\max(\alpha z,z)$ | $1$ o $\alpha$ |
| GELU | $z\,\Phi(z)\approx0.5z\big(1+\tanh[\sqrt{2/\pi}(z+0.044715z^3)]\big)$ | — |
| SiLU/Swish | $z\,\sigma(z)$ | $\sigma+z\sigma(1-\sigma)$ |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ | $s_i(\delta_{ij}-s_j)$ |

Sigmoide/tanh saturan → gradientes que se desvanecen. ReLU puede "morir" (neuronas siempre 0).

## 19.3 Perceptrón multicapa (MLP) y retropropagación ⭐⭐
**Forward:** $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$, $a^{(l)}=\sigma(z^{(l)})$, $a^{(0)}=x$.
**Backprop** (regla de la cadena de atrás hacia delante):
$$\delta^{(L)}=\nabla_{a^{(L)}}L\odot\sigma'(z^{(L)}),\qquad\delta^{(l)}=\big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot\sigma'(z^{(l)})$$
$$\frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}\,a^{(l-1)\top},\qquad\frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}$$
**Atajo clave:** salida softmax + entropía cruzada ⇒ $\delta^{(L)}=\hat y-y$. Con minibatch de tamaño $B$ las operaciones son matriciales: $\delta\in\mathbb R^{B\times n_l}$, $\nabla W=\frac1B a^\top\delta$.
```
for época: for batch:
    forward guardando z[l], a[l]
    δ = (ŷ - y)                       # softmax + CE
    for l = L..1: dW[l]=a[l-1]ᵀ δ / B; db[l]=sum(δ)/B; δ = (δ W[l]ᵀ) ⊙ σ'(z[l-1])
    actualizar parámetros con el optimizador
```
**Verifica siempre con comprobación numérica de gradientes** (13.1). Este es el ejercicio más formativo de toda la enciclopedia: impleméntalo solo con NumPy.

## 19.4 Funciones de pérdida ⭐
- **MSE:** $\frac1n\sum(y-\hat y)^2$, gradiente $\frac2n(\hat y-y)$.
- **Entropía cruzada (categórica):** $-\sum_ky_k\log\hat y_k$.
- **Huber:** cuadrática para $|e|\le\delta$, lineal fuera (robusta a *outliers*).
- **Hinge**, **focal loss** $-(1-p_t)^\gamma\log p_t$ (desbalanceo), **contrastiva / triplet** $\max(0,\ d(a,p)-d(a,n)+m)$, **InfoNCE** $-\log\frac{e^{s(q,k^+)/\tau}}{\sum_ke^{s(q,k)/\tau}}$.

## 19.5 Inicialización de pesos ⭐⭐
Mantener la varianza de activaciones y gradientes estable entre capas.
**Xavier/Glorot** (tanh, sigmoide): $\mathrm{Var}(W)=\frac2{n_{in}+n_{out}}$. **He** (ReLU): $\mathrm{Var}(W)=\frac2{n_{in}}$. Sesgos a 0. Nunca todos los pesos iguales (simetría).

## 19.6 Normalización: BatchNorm y LayerNorm ⭐⭐
**BatchNorm:** por canal, $\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, $y=\gamma\hat x+\beta$. En inferencia usa medias móviles. Backprop: pasa por $\mu_B,\sigma_B^2$ (gradiente completo, no ignorar dependencias).
**LayerNorm:** normaliza sobre las características de **cada muestra** (independiente del batch): estándar en Transformers. **RMSNorm:** $\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\gamma$.

## 19.7 Regularización: Dropout y Weight Decay ⭐⭐
**Dropout:** en entrenamiento, anular cada activación con prob. $p$ y reescalar por $\frac1{1-p}$ (*inverted dropout*); en inferencia, no hacer nada. **Weight decay** (L2): $\theta\leftarrow\theta(1-\eta\lambda)-\eta g$. Otros: *early stopping*, aumento de datos, *label smoothing*, *mixup*.

## 19.8 Convolución (CNN) ⭐⭐⭐
**Operación (correlación cruzada):** $y[i,j,f]=\sum_{c}\sum_{u,v}W[u,v,c,f]\;x[si+u,\,sj+v,\,c]+b_f$.
**Tamaño de salida:** $n_{out}=\Big\lfloor\dfrac{n+2p-k}{s}+1\Big\rfloor$ ($k$ = núcleo, $p$ = relleno, $s$ = paso).
**Parámetros:** $k^2\,C_{in}C_{out}$ (+ sesgos), compartidos por posición.
**Retropropagación:** $\partial L/\partial W$ = correlación entre la entrada y el gradiente de salida; $\partial L/\partial x$ = **convolución completa** del gradiente con el núcleo girado 180°.
**Implementación eficiente:** *im2col* (convertir parches en columnas y hacer un único GEMM). **Pooling:** max (gradiente solo al máximo) o promedio. **Conv 1×1**, **depthwise separable** (MobileNet), **dilatada**, **transpuesta** (upsampling).

## 19.9 Conexiones residuales y arquitecturas CNN ⭐⭐⭐
**ResNet:** $y=\mathcal F(x)+x$. El gradiente fluye por la identidad ($\partial y/\partial x=I+\partial\mathcal F/\partial x$), permitiendo cientos de capas. Hitos: LeNet → AlexNet → VGG → Inception → **ResNet** → DenseNet → EfficientNet → **ConvNeXt** → ViT (ver 19.15).

## 19.10 RNN y BPTT ⭐⭐⭐
$h_t=\tanh(W_hh_{t-1}+W_xx_t+b)$, $y_t=W_yh_t$. **Retropropagación a través del tiempo:** el gradiente incluye productos $\prod_k W_h^\top\mathrm{diag}(\tanh')$ → **se desvanece o explota** ($\rho(W_h)<1$ o $>1$). Mitigación: *gradient clipping* $g\leftarrow g\cdot\min(1,\tau/\|g\|)$, y puertas (LSTM/GRU).

## 19.11 LSTM ⭐⭐⭐
$$\begin{aligned}f_t&=\sigma(W_f[h_{t-1},x_t]+b_f)&&\text{(olvido)}\\ i_t&=\sigma(W_i[h_{t-1},x_t]+b_i)&&\text{(entrada)}\\ \tilde c_t&=\tanh(W_c[h_{t-1},x_t]+b_c)\\ c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t&&\text{(celda)}\\ o_t&=\sigma(W_o[h_{t-1},x_t]+b_o),\quad h_t=o_t\odot\tanh(c_t)\end{aligned}$$
La celda $c_t$ es un "carril" casi lineal que preserva el gradiente. **GRU:** $z_t=\sigma(W_z[h_{t-1},x_t])$, $r_t=\sigma(W_r[h_{t-1},x_t])$, $\tilde h_t=\tanh(W[r_t\odot h_{t-1},x_t])$, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$.

## 19.12 Seq2Seq y atención de Bahdanau ⭐⭐⭐
Codificador-decodificador. Atención aditiva: $e_{ti}=v^\top\tanh(W_ss_{t-1}+W_hh_i)$, $\alpha_{ti}=\mathrm{softmax}_i(e_{ti})$, contexto $c_t=\sum_i\alpha_{ti}h_i$. Decodificación: *greedy*, **beam search** (mantener las $B$ mejores hipótesis por log-prob.), muestreo con temperatura / top-k / top-p.

## 19.13 Atención escalada (Scaled Dot-Product) ⭐⭐⭐
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V$$
$Q=XW_Q,\ K=XW_K,\ V=XW_V$. El factor $\sqrt{d_k}$ evita que los productos escalares (varianza $d_k$) saturen el softmax. $M$ = máscara causal ($-\infty$ sobre el futuro). Coste $O(n^2d)$, memoria $O(n^2)$.
**Multi-head:** $h$ cabezas con $d_k=d/h$, concatenar y proyectar con $W_O$.

## 19.14 Transformer ⭐⭐⭐
**Bloque (pre-norm):**
$$x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x)),\qquad x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$$
$\mathrm{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2$, con dimensión intermedia $\approx4d$ (o SwiGLU).
**Codificación posicional:** sinusoidal $PE_{(pos,2i)}=\sin\frac{pos}{10000^{2i/d}}$, $PE_{(pos,2i+1)}=\cos\frac{pos}{10000^{2i/d}}$; o aprendida; o **RoPE** (rotar $q,k$ en pares de dimensiones por ángulo $pos\cdot\theta_i$); o ALiBi.
**Entrenamiento (LM autorregresivo, estilo GPT):** pérdida $-\sum_t\log p(x_t|x_{<t})$ (*teacher forcing*, máscara causal). **BERT:** modelado de lenguaje enmascarado, atención bidireccional. **Encoder-decoder (T5):** *cross-attention* donde $K,V$ vienen del codificador.
**Eficiencia:** KV-cache en inferencia, **FlashAttention** (cálculo por bloques sin materializar $n\times n$), GQA/MQA (compartir $K,V$), cuantización, **MoE** (enrutar cada token a $k$ de $E$ expertos).
**Ajuste fino eficiente (LoRA):** $W'=W+\frac\alpha rBA$, con $B\in\mathbb R^{d\times r}$, $A\in\mathbb R^{r\times d}$, $r\ll d$; solo se entrenan $A,B$.

## 19.15 Vision Transformer (ViT) ⭐⭐⭐
Trocear la imagen en parches $P\times P$, aplanar y proyectar linealmente a tokens, añadir `[CLS]` y posición, pasar por un Transformer. Número de tokens: $\frac{HW}{P^2}$.

## 19.16 Tokenización: BPE ⭐⭐
Partir del vocabulario de bytes/caracteres; repetir: contar pares adyacentes en el corpus, **fusionar el más frecuente** en un nuevo símbolo, hasta alcanzar el tamaño de vocabulario. Codificar aplicando las fusiones en orden. Alternativas: WordPiece, Unigram LM (SentencePiece).

## 19.17 Word2Vec y embeddings ⭐⭐
**Skip-gram con muestreo negativo:** maximizar $\log\sigma(u_o^\top v_c)+\sum_{k=1}^K\log\sigma(-u_{w_k}^\top v_c)$ con $w_k\sim P(w)^{3/4}$. **GloVe:** factoriza $\log X_{ij}$ ponderado. Similitud por coseno; analogías como aritmética vectorial.

## 19.18 Autoencoders ⭐⭐
Codificador $z=f(x)$ + decodificador $\hat x=g(z)$; minimizar $\|x-\hat x\|^2$. Variantes: *denoising* (entrada corrupta), *sparse* (L1 sobre $z$), contractivo.

## 19.19 Autoencoder variacional (VAE) ⭐⭐⭐
Codificador $q_\phi(z|x)=\mathcal N(\mu_\phi,\mathrm{diag}\,\sigma_\phi^2)$, decodificador $p_\theta(x|z)$, prior $\mathcal N(0,I)$. Maximizar el ELBO:
$$\mathcal L=E_{q}[\log p_\theta(x|z)]-D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)$$
**KL cerrada:** $-\frac12\sum_j\big(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\big)$.
**Truco de reparametrización:** $z=\mu+\sigma\odot\epsilon$, $\epsilon\sim\mathcal N(0,I)$ → permite retropropagar a través del muestreo. Generar: muestrear $z\sim\mathcal N(0,I)$ y decodificar. **VQ-VAE:** espacio latente discreto con *codebook*.

## 19.20 GAN ⭐⭐⭐
Generador $G(z)$ y discriminador $D(x)$ en juego minimax:
$$\min_G\max_D\ E_{x}[\log D(x)]+E_z[\log(1-D(G(z)))]$$
En la práctica, el generador maximiza $\log D(G(z))$ (*non-saturating*). Alterna pasos de $D$ y $G$. **Problemas:** inestabilidad, colapso de modos. **WGAN:** distancia de Wasserstein con $D$ 1-Lipschitz (penalización de gradiente $\lambda(\|\nabla D\|-1)^2$). Condicional (cGAN), StyleGAN, CycleGAN.

## 19.21 Modelos de difusión (DDPM) ⭐⭐⭐
**Proceso directo:** $q(x_t|x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_tI)$. Con $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$:
$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal N(0,I)$$
**Entrenamiento:** elegir $t$ al azar, ruido $\epsilon$, minimizar $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ (red U-Net o Transformer).
**Muestreo (DDPM):** $x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big)+\sigma_tz$.
**Mejoras:** DDIM (determinista, menos pasos), *classifier-free guidance* $\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+s\,(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$, difusión latente (Stable Diffusion), *flow matching*.

## 19.22 Redes neuronales de grafos (GNN) ⭐⭐⭐
**GCN:** $H^{(l+1)}=\sigma\big(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\big)$, $\tilde A=A+I$. **GAT:** coeficientes de atención entre vecinos. Esquema general: *message passing* — agregar mensajes de vecinos (suma/media/máx) y actualizar el estado del nodo.

## 19.23 Aprendizaje contrastivo y autosupervisado ⭐⭐⭐
**SimCLR / MoCo:** dos vistas aumentadas del mismo ejemplo = positivos, el resto del lote = negativos; pérdida InfoNCE (19.4). **CLIP:** alinear imagen y texto con InfoNCE simétrica sobre la matriz de similitudes del batch. **BYOL / DINO:** sin negativos (profesor con media móvil exponencial). **MAE:** reconstruir parches enmascarados.

## 19.24 Destilación de conocimiento ⭐⭐
El alumno imita las probabilidades suaves del profesor: $L=(1-\alpha)\,CE(y,p_s)+\alpha\,T^2\,KL\big(\mathrm{softmax}(z_t/T)\,\|\,\mathrm{softmax}(z_s/T)\big)$.

## 19.25 Redes de Hopfield, Boltzmann y otras clásicas ⭐⭐⭐
**Hopfield:** memoria asociativa, $W=\frac1N\sum_\mu\xi^\mu\xi^{\mu\top}$ (diag = 0); actualización $s_i\leftarrow\mathrm{sign}(\sum_jW_{ij}s_j)$ disminuye la energía $E=-\frac12s^\top Ws$. **RBM:** $P(v,h)\propto e^{-E}$, entrenada con *contrastive divergence*. **SOM (Kohonen):** $w_i\leftarrow w_i+\eta\,h(i,i^*)(x-w_i)$ sobre una rejilla de vecindad.

## 19.26 Entrenamiento práctico: lista de comprobación ⭐⭐
1. Sobreajustar un mini-lote (10 muestras) hasta pérdida ≈ 0 — si no, hay un *bug*.
2. Pérdida inicial esperada: $\approx\ln C$ en clasificación de $C$ clases.
3. Comprobar gradientes numéricamente en un modelo pequeño.
4. Normalizar entradas; inicialización He/Xavier; barajar datos.
5. Buscar la tasa de aprendizaje primero (escala logarítmica); *warmup* + coseno.
6. Vigilar normas de gradientes (*clipping*) y activaciones (NaN/Inf).
7. Separar *train/val/test* sin fugas; fijar semillas.
8. Precisión mixta (bf16/fp16) con *loss scaling*; acumulación de gradientes para lotes grandes.

---

# 20. Hoja de ruta de implementación

Orden recomendado para ir codificando la enciclopedia desde cero:

| Etapa | Qué implementar | Capítulos |
|---|---|---|
| **1. Cimientos** | Búsqueda binaria, ordenaciones básicas, merge, quick, heap | 2, 3, 4.2 |
| **2. Estructuras** | BST, hash, trie, union-find, Fenwick | 4 |
| **3. Grafos** | BFS, DFS, Dijkstra, Kruskal, topológico, flujo | 5 |
| **4. DP y strings** | Mochila, LCS, edición, KMP | 6, 7 |
| **5. Numéricos** | Euclides, criba, Newton, Gauss, FFT | 8 |
| **6. Optimización** | GD, SGD, Adam, SA, GA | 13, 14 |
| **7. ML clásico (NumPy puro)** | OLS, logística, k-NN, k-means, árbol, PCA, Naive Bayes | 15, 16 |
| **8. Probabilístico y RL** | HMM-Viterbi, Metropolis, Q-learning | 17, 18 |
| **9. Deep Learning (solo NumPy)** | MLP con backprop → CNN → LSTM | 19.1-19.11 |
| **10. Deep Learning moderno** | Atención → Transformer mini (estilo "nanoGPT") → VAE → difusión | 19.13-19.21 |

**Consejos para el repositorio:**
- Un archivo por algoritmo con **tests** (comparar con la librería de referencia: `sorted`, `scipy`, `sklearn`).
- Cada implementación: docstring con la complejidad y un enlace a su sección de esta enciclopedia.
- Incluir una carpeta `benchmarks/` para medir empíricamente $O(\cdot)$ y verificar curvas de crecimiento.
- Estructura sugerida: `algoritmos/{busqueda,ordenacion,grafos,dp,strings,numericos,geometria,ml,dl,rl}/`.

## Bibliografía recomendada
- Cormen, Leiserson, Rivest, Stein — *Introduction to Algorithms* (CLRS).
- Sedgewick & Wayne — *Algorithms*.
- Skiena — *The Algorithm Design Manual*.
- Bishop — *Pattern Recognition and Machine Learning*.
- Murphy — *Probabilistic Machine Learning* (libros 1 y 2).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*.
- Sutton & Barto — *Reinforcement Learning: An Introduction*.
- Goodfellow, Bengio, Courville — *Deep Learning*.
- Nocedal & Wright — *Numerical Optimization*.
- Russell & Norvig — *Artificial Intelligence: A Modern Approach*.
- Papers clave: *Attention Is All You Need* (2017), *Adam* (2014), *ResNet* (2015), *DDPM* (2020), *DQN* (2015), *PPO* (2017).

---

*Documento vivo: cada algoritmo puede ampliarse con implementaciones, casos de prueba y visualizaciones.*