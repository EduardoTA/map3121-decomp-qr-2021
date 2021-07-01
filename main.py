# Nome: Eduardo Tadashi Asato 		    nusp: 10823810
# Nome: Gustavo Santos da Silva 		nusp: 10432387

import numpy as np
import math as math

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


class CalculaAutovalsAutovecs:
    # Este método faz o cálculo dos cossenos (forma estável)
    def __calcula_c(self, alfa, beta, i):
        if abs(alfa[i]) > abs(beta[i]):
            tau = -beta[i] / alfa[i]
            return 1 / math.sqrt(1 + tau * tau)
        else:
            tau = -alfa[i] / beta[i]
            return tau / math.sqrt(1 + tau * tau)

    # Este método faz o cálculo dos senos (forma estável)
    def __calcula_s(self, alfa, beta, i):
        if abs(alfa[i]) > abs(beta[i]):
            tau = -beta[i] / alfa[i]
            return tau / math.sqrt(1 + tau * tau)
        else:
            tau = -alfa[i] / beta[i]
            return 1 / math.sqrt(1 + tau * tau)

    # Este método faz o cálculo de mu
    # m: o tamanho-1 da submatriz onde o método é aplicado
    def __calcula_mu(self, alfa, beta, m, deslocamentos):
        # O if abaixo detecta se foram solicitados deslocamentos espectrais
        if deslocamentos:
            # Precisamos obter primeiramente dk.
            dk = (alfa[m - 1] - alfa[m]) / 2
            mu = alfa[m] + dk - np.sign(dk) * math.sqrt(dk ** 2 + beta[m - 1] ** 2)
            return mu
        else:
            return 0

    # Este método faz a normalização de vector
    def __normaliza(self, vector):
        soma = 0
        for i in range(0, len(vector)):
            soma = soma + vector[i] ** 2
        return vector / math.sqrt(soma)

    # Este é o método público desta classe
    def AutovalAutovec(self, alfa, beta, epsilon, deslocamentos, V):
        c = np.zeros(alfa.shape[0] - 1) # Vetor que armazena os valores de c em cada subiteração i da decomposição
                                        # QR da k-ésima iteração
        s = np.zeros(alfa.shape[0] - 1) # Vetor que armazena os valores de s em cada subiteração i da decomposição
                                        # QR da k-ésima iteração
        mu = 0

        # De acordo com o for, esses 3 vetores mudam de função
        Diagonal_principal = np.copy(alfa)
        Subdiagonal_superior = np.copy(beta)
        Subdiagonal_inferior = np.copy(beta)

        # Este vetor é usado para armazenar o valor intermediário de V para que ele seja atribuído no final
        temp_V = np.copy(V)

        # O algoritmo itera sobre matriz m+1 por m+1, que vai diminuindo à medida que os autovalores são encontrados
        m = alfa.shape[0] - 1

        k = 0 # Contador de iterações

        while (m >= 0):
            # Estes quatro vetores temp são usados como temporários para armazenar as células da
            # matriz que são atualizadas entre subiterações dos três fors
            temp_diagonal_principal = np.zeros(2)
            temp_subdiagonal = np.zeros(2)

                # Quando multiplicamos o subresultado de V(k+1) por Qi(k), só são modificadas
                # duas colunas do subresultado, aqui estão os dois temporários das duas colunas modificadas
            temp_V_coluna_esquerda = np.zeros(alfa.shape[0])
            temp_V_coluna_direita = np.zeros(alfa.shape[0])

            # O if só é executado se a submatriz é pelo menos 2x2 e o último beta não puder ser zerado
            if (m > 0 and abs(Subdiagonal_inferior[m - 1]) >= epsilon):
                if (k > 0):
                    mu = self.__calcula_mu(Diagonal_principal, Subdiagonal_inferior, m, deslocamentos)
                    Diagonal_principal = Diagonal_principal - mu
                # Este for faz a decomposição QR
                #
                # Este for realiza m subiterações i
                # A cada subiteração o subresultado de R(k) é multiplicado por Qi(k)
                # Até no final ter a própria R(k)
                #
                # Funções dos seguinte vetores neste for:
                # Diagonal_principal: Vetor que armazena a diagonal principal do subresultado de R(k),
                # e. no final do for, da própria R
                #
                # Subdiagonal_superior: Vetor que armazena a 1ª subdiagonal acima da diagonal principal do
                # subresultado de R(k), e. no final do for, da própria R(k)
                #
                # Subdiagonal_inferior: Vetor que armazena a 1ª subdiagonal abaixo da diagonal principal do
                # subresultado de R(k), e. no final do for, da própria R(k)
                #
                #
                #
                # Otimizações implementadas neste for:
                # 1. Só são armazenados os cossenos e senos, e não a matriz Qi(k) inteira
                # 2. O elemento da subiteração i do subresultado de R(k) de posição (i,i+1) é zerado automaticamente
                # 3. Só se calcula os valores da diagonal principal e das duas subdiagonais, imediatamente abaixo e
                # acima da principal, pois os elementos das outras diagonais são zerados e não são usados
                # no cálculo de A(k+1)
                # 4. Entre cada subiteração somente 5 células importantes são alteradas no subresultado de R(k),
                # exceto na última iteração
                for i in range(0, m):
                    # Calcula c e s da subiteração i da decomposição QR #Otimização 1
                    c[i] = self.__calcula_c(Diagonal_principal, Subdiagonal_inferior, i)
                    s[i] = self.__calcula_s(Diagonal_principal, Subdiagonal_inferior, i)

                    # Calcula os novos valores das duas células atualizadas da diagonal principal
                    # do subresultado de R(k)
                    temp_diagonal_principal[0] = c[i] * Diagonal_principal[i] - s[i] * Subdiagonal_inferior[i]
                    temp_diagonal_principal[1] = s[i] * Subdiagonal_superior[i] + c[i] * Diagonal_principal[i + 1]

                    # Calcula os novos valores das duas células atualizadas da 1ª subdiagonal acima
                    # da diagonal principal do subresultado de R(k)
                    temp_subdiagonal[0] = c[i] * Subdiagonal_superior[i] - s[i] * Diagonal_principal[i + 1]
                    if (i != m - 1):
                        # Não existe Subdiagonal_superior[i+1] na última subiteração, por isso este if
                        temp_subdiagonal[1] = c[i] * Subdiagonal_superior[i + 1]

                    # Atualiza o subresultado com os valores temporários
                    Subdiagonal_inferior[i] = 0.0  # Otimização 2
                    Diagonal_principal[i] = np.copy(temp_diagonal_principal[0])
                    Diagonal_principal[i + 1] = np.copy(temp_diagonal_principal[1])
                    Subdiagonal_superior[i] = np.copy(temp_subdiagonal[0])

                    if (i != m - 1):
                        Subdiagonal_superior[i + 1] = np.copy(temp_subdiagonal[1])

                # Este for faz o cálculo de A(k+1)
                #
                # Este for realiza m subiterações i
                # A cada subiteração o subresultado de A(k+1) é multiplicado por Qi(k)T
                # Até no final ter a própria A(k+1)
                #
                # Neste for
                # Diagonal_principal: Vetor que armazena a diagonal principal do subresultado de A(k+1),
                # e. no final do for, da própria A(k+1)
                #
                # Subdiagonal_superior: Vetor que armazena a 1ª subdiagonal acima da diagonal principal
                # do subresultado de A(k+1), e. no final do for, da própria A(k+1)
                #
                # Subdiagonal_inferior: Vetor que armazena a 1ª subdiagonal abaixo da diagonal principal
                # do subresultado de A(k+1), e. no final do for, da própria A(k+1)
                #
                #
                #
                # Otimizações implementadas neste for:
                # 1. Como a matriz A(k+1) termina simétrica, então só calculamos o valor para a
                # diagonal imediatamente abaixo da principal
                # 2. Como as multiplicações por Qi(k)T só modificam duas colunas do subresultado de A(k+1),
                # então só é necessário calcular 3 células do subresultado
                for i in range(0, m):
                    # Calcula os novos valores das duas células atualizadas
                    # da diagonal principal do subresultado de A(k+1)
                    temp_diagonal_principal[0] = c[i] * Diagonal_principal[i] - s[i] * Subdiagonal_superior[i]
                    temp_diagonal_principal[1] = c[i] * Diagonal_principal[i + 1]

                    # Calcula o novo valos da célula atualizada da 1ª subdiagonal abaixo
                    # da diagonal principal do subresultado de A(k+1)
                    temp_subdiagonal[0] = -s[i] * Diagonal_principal[i + 1]  # Otimização 1

                    # Atualiza o subresultado com os valores temporários
                    Diagonal_principal[i] = np.copy(temp_diagonal_principal[0])
                    Diagonal_principal[i + 1] = np.copy(temp_diagonal_principal[1])
                    Subdiagonal_inferior[i] = np.copy(temp_subdiagonal[0])

                    # Como A(k+1) é simétrica, não precisamos calcular a subdiagonal superior #Otimização 1
                    Subdiagonal_superior[i] = np.copy(temp_subdiagonal[0])

                # Faz A(k+1) = R(k)*Q(k)+muk*I
                if (k > 0):
                    Diagonal_principal = Diagonal_principal + mu

                # Este for faz a multiplicação V(k+1)=V(k)*Q(k)T, e armazena o resultado em temp_V
                #
                # Este for realiza m subiterações subiteracao
                # A cada subiteração o subresultado de V(k+1) é multiplicado por Qi(k)T
                # Até no final ter a própria V(k+1)
                #
                # Otimizações implementadas neste for:
                # 1. Como as multiplicações por Qi(k)T só modificam duas colunas do subresultado de V(k+1),
                # então só é necessário calcular 2 colunas do subresultado
                for subiteracao in range(0, m):
                    for linha in range(0, alfa.shape[0]):
                        temp_V_coluna_esquerda[linha] = temp_V[linha, subiteracao] * c[subiteracao]\
                                                        - temp_V[linha, subiteracao + 1] * s[subiteracao]
                        temp_V_coluna_direita[linha] = temp_V[linha, subiteracao + 1] * c[subiteracao]\
                                                       + temp_V[linha, subiteracao] * s[subiteracao]

                        temp_V[linha, subiteracao] = temp_V_coluna_esquerda[linha]
                        temp_V[linha, subiteracao + 1] = temp_V_coluna_direita[linha]

                k = k + 1


            # Este if faz a eliminação de beta se este for menor que epsilon e diminui o escopo (m=m-1)
            # do algoritmo, para que ele passe a trabalhar com a submatriz
            if (abs(Subdiagonal_inferior[m - 1]) < epsilon):
                Subdiagonal_inferior[m - 1] = 0.0
                Subdiagonal_superior[m - 1] = 0.0
                m = m - 1

        # __normaliza autovetores
        for j in range(0, alfa.shape[0]):
            temp_V[:, j] = self.__normaliza(temp_V[:, j])
        return (Diagonal_principal, temp_V, k)

class EstimadorDeErros:
    def __normaliza(self, vector):
        soma = 0
        for i in range(0, len(vector)):
            soma = soma + vector[i] ** 2
        return vector / math.sqrt(soma)

    def EstimativasErroAnalitico(self, n, Lambda, V):
        # Calcula os autovalores reais, pela fórmula analítica, e ordena eles do maior para o menor
        autovalores_reais = 2 * (1 - np.cos(np.arange(n, 0, -1) * math.pi / (n + 1)))
        autovalores_reais = np.flip(np.sort(autovalores_reais))

        autovalores_obtidos = np.copy(np.flip(np.sort(Lambda)))

        erros = np.zeros(n)

        # Faz a comparação entre os autovalores reais e os obtidos pelo método QR
        print('Autovalor obtido | Autovalor real | erro')
        for i in range(0, n):
            erros[i] = math.sqrt(math.pow(autovalores_obtidos[i] - autovalores_reais[i], 2))
            print('{0:12.10f}       {1:12.10f}     {2}'.format(autovalores_obtidos[i], autovalores_reais[i],erros[i]))
        print('-----------------------------')
        print('Erro máx: ', np.max(erros))
        print('\n')

        # Calcula os autovetores reais, pela fórmula analítica, montando uma matriz semelhante à V
        autovetores_reais = np.zeros((n, n))
        for j in range(0, n):
            autovetores_reais[:, j] = np.sin(np.arange(1, n + 1, 1) * (n - j) * math.pi / (n + 1))
            autovetores_reais[:, j] = self.__normaliza(autovetores_reais[:, j])

        # Faz a comparação entre os autovetores reais e os obtidos pelo método QR
        print('Erro obtido fazendo max(abs(Autovetores_reais-Autovetores_obtidos)) = {0}\n\n'
              .format(np.max(abs(autovetores_reais - V))))

    def EstimativasErroGerais(self, n, alfa, beta, Lambda, V):
        # Faz uma estimativa de erro = max(abs(matmul(Q,QT)-I))
        erro = np.max(np.abs(np.matmul(V, np.transpose(V)) - np.identity(n)))
        print('max(abs(matmul(Q,QT)-I)) = {0}'.format(erro))

        # Faz uma estimativa de erro = max(abs(matmul(A,Q)-matmul(Q,L)))
        A = np.zeros((n, n))

        for i in range(0, n):
            A[i, i] = alfa[i]
        for i in range(0, n - 1):
            A[i, i + 1] = beta[i]
            A[i + 1, i] = beta[i]

        L = np.zeros((n, n))

        for i in range(0, n):
            L[i, i] = Lambda[i]

        erro = np.max(np.abs(np.matmul(A, V) - np.matmul(V, L)))
        print('max(abs(matmul(A,Q)-matmul(Q,L))) = {0}\n'.format(erro))

        # Faz uma estimativa de erro comparando (A*v)/v, com os autovalores obtidos
        erros = np.zeros(n)
        for i in range(0, n):
            erros[i] = math.sqrt(math.pow(np.mean(np.divide(np.matmul(A, V[:, i]), V[:, i])) - Lambda[i], 2))
            print('Autovalor obtido fazendo matdiv(matmul(A,v),v): {0:12.10f},'
                  '  Autovalor obtido pelo método QR: {1:12.10f}, erro: {2}'
                  .format(np.mean(np.divide(np.matmul(A, V[:, i]), V[:, i])), Lambda[i], erros[i]))
        print('Erro máximo = {0}\n'.format(np.max(erros)))


class Interface:
    alfa = np.zeros(1) # Valores da diagonal principal da matriz tridiagonal
    beta = np.zeros(1) # Valores de uma das subdiagonais, abaixo da principal, da matriz tridiagonal

    n = 0 # Tamanho da matriz

    usar_caso_analitico = False # Booleano que indica se deve ser criados alfa e beta para caso com solução analítica

    # Booleano que indica se o método QR deve ser realizado com deslocamentos espectrais
    usar_deslocamentos_espectrais = False

    epsilon = 0.000001 # Tolerância para determinação da condição de parada
    V = np.identity(alfa.shape[0]) # Matriz onde os autovetores serão armazenados
    Lambda = 0 # Vetor onde os autovalores serão armazenados
    k = 0 # n° de iterações que foram necessárias para convergir

    resultados = () # Tupla onde os resultados são armazenados

    calculadora = CalculaAutovalsAutovecs()
    estimadorDeErros = EstimadorDeErros()

    # Para o modo de resposta temporal de sistema massa-mola
    n_massas = 0 # n° de massas do sistema massa-mola
    massa = 0.0 # massa das massas do sistema
    X0 = np.zeros(n_massas) # Vetor com as condições iniciais do problema
    Y0 = np.zeros(n_massas) # Vetor com as condições iniciais do problema em Y(t)
    ki = np.zeros(n_massas+1) # Vetor com as constantes de mola do sistema

    # Esta é a primeira tela vista pelo usuário, a tela 0
    def promptTelaInicial(self):
        # Menu de seleção inicial
        print('Selecione uma opção')
        print('(0) para calcular autovalores e autovetores de matriz tridiagonal')
        print('(1) para obter resposta de sistema massa-mola')
        print('(qualquer outra) para sair')
        selecao = input('=')

        if(selecao == '0'):
            self.__promptCalcularAutovalsAutovecs()
            print('->Saindo...')
            self.promptTelaInicial()
        elif(selecao == '1'):
            self.__promptMassaMola()
            print('->Saindo...')
            self.promptTelaInicial()
        else:
            print('->Saindo...')
            return

    # Caso a opção 0 seja escolhida na tela 0
    def __promptCalcularAutovalsAutovecs(self):
        print('->Selecionado modo de cálculo de autovalores e autovetores') # Confirmação de seleção

        # Bloco try-catch garante que o programa não irá ter erro de runtime caso entrada inválida
        try:
            self.n = int(input('Tamanho da matriz A: n = ')) # Recebe a entrada de usuário do atributo n

            print('Resettando matriz V')
            self.V = np.identity(self.n)

            # Pergunta ao usuário se quer que o programa gere automaticamente alfa e beta para caso analítico
            if(input('Usar caso com solução analítica? (y/n)') == 'y'):
                self.usar_caso_analitico = True
                self.alfa = 2*np.ones(self.n)
                self.beta = -1*np.ones(self.n-1)
            else:
                # Caso não queira, alfa e beta devem ser digitados manualmente
                print('Digitar os vetores da seguinte forma (n=3): 1 2 3')
                self.alfa = np.array(list(map(float, input("alfa = ").strip().split()))[:self.n])
                self.beta = np.array(list(map(float, input("beta = ").strip().split()))[:self.n - 1])

            # Solicita ao usuário se deseja usar deslocamentos espectrais
            if(input('Usar deslocamentos espectrais? (y/n) ') == 'y'):
                self.usar_deslocamentos_espectrais = True
            else:
                self.usar_deslocamentos_espectrais = False

            self.epsilon = float(input('epsilon: ep = ')) # Recebe a entrada de tolerância do usuário

            # Se as dimenões de alfa e beta estejam erradas, raise exception
            if(self.alfa.shape[0] != self.n and self.beta.shape[0] != self.n-1):
                raise Exception()
        except:
            # Se erro de digitação for detectado, voltar para tela 0
            print('Erro de digitação')
            return

        # Aplica o método de obtenção de autovalores e autovetores usando decomposição QR
        self.resultados = self.calculadora.AutovalAutovec(self.alfa, self.beta, self.epsilon,
                                                          self.usar_deslocamentos_espectrais, self.V)
        # Distribui os resultados na tupla para Lambda, V e k
        self.Lambda = self.resultados[0]
        self.V = self.resultados[1]
        self.k = self.resultados[2]

        # Troca para a tela 1
        self.__promptTelaResultadosCalculoAutovalsAutovecs()

    # Imprime a tela 1
    def __promptTelaResultadosCalculoAutovalsAutovecs(self):

        # Número de iterações necessárias
        print('Foram necessárias {0:d} iterações'.format(self.k))

        # Menu de seleção
        print('Selecione uma opção')
        print('(0) Obter autovalores e autovetores')
        print('(1) Cálculo de estimativas de erro')
        print('(qualquer outra) para sair')
        selecao = input('=')

        if(selecao == '0'):
            self.__mostrarAutovalsAutovecs() # Troca para a tela 2
            self.__promptTelaResultadosCalculoAutovalsAutovecs()
        elif(selecao == '1'):
            self.__promptEstimativasDeErro() # Troca para tela 3 ou 4
            self.__promptTelaResultadosCalculoAutovalsAutovecs()
        else:
            return

    # Imprime a tela 2
    def __mostrarAutovalsAutovecs(self):
        for i in range(0, self.alfa.shape[0]):
            # Configuração de impressão para ter mais dígitos
            np.set_printoptions(formatter={'float': '{: 12.10f}'.format})

            # Impressão dos autovals/autovecs
            print('Autovalor: {0:12.10f}, Autovetor:'.format(self.Lambda[i]), self.V[:, i])

    # Imprime tela 3 ou 4
    def __promptEstimativasDeErro(self):
        # Por padrão imprime a tela 3,
        # porém se for detectada matriz com solução analítica, tela 4
        if (np.array_equal(self.alfa, 2 * np.ones(self.n)) and np.array_equal(self.beta, -1 * np.ones(self.n - 1))):
            # Menu de seleção
            print('Selecione uma opção')
            print('(0) Estimativas de erro gerais')
            print('(1) Estimativas de erro comparando com o resultado analítico')
            print('(qualquer outra) para sair')
            selecao = input('=')

            if(selecao == '0'):
                # Se as estimativas de erro gerais forem selecionadas,
                # imprime a tela 3 com as estimativas de erro para caso geral
                self.estimadorDeErros.EstimativasErroGerais(self.n, self.alfa, self.beta, self.Lambda, self.V)
                self.__promptEstimativasDeErro()
            elif(selecao == '1'):
                # Se as estimativas de erro gerais forem selecionadas,
                # imprime a tela 5 com as estimativas de erro para caso com solução analítica
                self.estimadorDeErros.EstimativasErroAnalitico(self.n, self.Lambda, self.V)
                self.__promptEstimativasDeErro()
            else:
                return
        else:
            # Por padrão imprime a tela 3
            self.estimadorDeErros.EstimativasErroGerais(self.n, self.alfa, self.beta, self.Lambda, self.V)
            return

    # Se for escolhida a opção 1 no menu principal, entra neste método
    def __promptMassaMola(self):
        print('->Selecionado modo de obtenção de resposta de sistema massa-mola') # Confirmação de seleção

        # Bloco try-catch garante que o programa não irá ter erro de runtime caso entrada inválida
        try:
            # Recebe a entrada de usuário de n° de massas do sistema
            self.n_massas = int(input('Número de massas: n_massas = '))

            # Reset de alguns atributos
            print('Resettando matriz V')
            self.V = np.identity(self.n_massas)
            self.X0 = np.zeros(self.n_massas)
            self.Y0 = np.zeros(self.n_massas)
            self.ki = np.zeros(self.n_massas+1)

            self.massa = float(input('Massa = ')) # Recebe a entrada de usuário de massa das massas do sistema

            # Solicita ao usuário se deseja usar deslocamentos espectrais
            if (input('Usar deslocamentos espectrais? (y/n) ') == 'y'):
                self.usar_deslocamentos_espectrais = True
            else:
                self.usar_deslocamentos_espectrais = False

            self.epsilon = float(input('epsilon: ep = '))
        except:
            # Se erro de digitação for detectado, voltar para tela 0
            print('Digitação errada')
            return

        # Troca para a tela 1'
        self.__promptMenuMassaMola()

    # Imprime no terminal a tela 1'
    def __promptMenuMassaMola(self):
        print('Selecione qual opção de constante de mola será usada:')
        print('(0): ki = (40+2i)')
        print('(1): ki = (40+2(-1)^i)')
        print('(2): Digitar as condições iniciais do problema (X0)')
        print('(3): Digitar os valores de constantes de mola')
        print('(4): GERAR RESULTADO')
        print('(qualquer outra) para sair')
        selecao = (input('='))

        if(selecao == '0'):
            # Se a opção 0 for selecionada,
            # as constantes de mola das molas são definidas seguindo a fórmula: ki = (40+2i)
            print('Seleção 0')
            self.ki = 40 + 2 * np.arange(1, self.n_massas + 2)
            self.__promptMenuMassaMola()
        elif(selecao == '1'):
            # Se a opção 1 for selecionada,
            # as constantes de mola das molas são definidas seguindo a fórmula: ki = (40+2(-1)^i)
            print('Seleção 1')
            self.ki = 40 + 2 * (-1) ** np.arange(1, self.n_massas + 2)
            self.__promptMenuMassaMola()
        elif(selecao == '2'):
            # Solicitar do usuário as condições iniciais dos x(t)
            print('Seleção 2')
            print('Digitar as condições iniciais (X0):')
            print('Digitar o vetor da seguinte forma: 1 2 3 4 5')

            try:
                self.X0 = np.array(list(map(float, input("X0 = ").strip().split()))[:self.n_massas])
                if (self.X0.shape[0] != self.n_massas):
                    raise Exception()
            except:
                print('Digitação errada')
            self.__promptMenuMassaMola()
        elif(selecao == '3'):
            # Caso o usuário deseje, pode digitar manualmente as constantes de mola das molas
            print('Seleção 3')
            print('Digitar as constantes de mola (ki):')
            print('Digitar o vetor da seguinte forma: 1 2 3 4 5')

            try:
                self.ki = np.array(list(map(float, input("ki = ").strip().split()))[:self.n_massas+1])
                print('ki=',self.ki)
                if (self.ki.shape[0] != self.n_massas+1):
                    raise Exception()
            except:
                print('Digitação errada')
            self.__promptMenuMassaMola()
        elif(selecao == '4'):
            # Aqui o programa gera os resultados

            self.alfa = np.ones(self.n_massas) # alfa é 'resettado' para ter n_massas de tamanho
            self.beta = np.ones(self.n_massas - 1) # beta é 'resettado' para ter n_massas-1 de tamanho
            self.V = np.identity(self.n_massas) # A matriz com autovetores é 'resettada' para identidade

            # Estes dois fors montam a matriz A para o problema do sistema massa-mola
            for i in range(0, self.n_massas):
                self.alfa[i] = (self.ki[i] + self.ki[i + 1]) / self.massa
            for i in range(0, self.n_massas - 1):
                self.beta[i] = -self.ki[i + 1] / self.massa

            # Aqui os autovalores e os autovetores da matriz A são calculados
            self.resultados = self.calculadora.AutovalAutovec(self.alfa, self.beta, self.epsilon,
                                                              self.usar_deslocamentos_espectrais, self.V)
            self.Lambda = self.resultados[0]
            self.V = self.resultados[1]

            self.__promptTelaResultadosMassaMola() # A tela 4' é chamada
            self.__promptMenuMassaMola()
        else:
            return

    # Este método imprime a tela 4'
    def __promptTelaResultadosMassaMola(self):
        print('Selecione uma opção:')
        print('(0): Gerar gráfico')
        print('(1): Imprimir frequências e modos naturais')
        selecao = (input('='))

        if(selecao == '0'):
            # Se a opção 0 for selecionada, gerar o gráfico, indo para a tela 5'
            self.__gerarGrafico()
            self.__promptTelaResultadosMassaMola()
        elif(selecao == '1'):
            # Se a opção 1 for selecionada, a tela 6' é impressa
            for i in range(0, self.n_massas):
                np.set_printoptions(formatter={'float': '{: 12.10f}'.format})

                # Aqui as frequências e os modos naturais são impressos
                print('freq: {0:12.10f}, Modo natural:'.format(math.sqrt(self.Lambda[i])), self.V[:,i])
            self.__promptTelaResultadosMassaMola()
        else:
            print('->Saindo...')
            return

    # Este método imprime a tela 5' e gera o gráfico
    def __gerarGrafico(self):
        # Este for gera Y(0) a partir de X(0)
        # Faz isto usando a relação Y(0)=Q*X(0)
        for i in range(0, self.n_massas):
            soma = 0
            for j in range(0, self.n_massas):
                soma = soma + np.transpose(self.V)[i, j] * self.X0[j]
            self.Y0[i] = soma

        # Esta função é usada por plt.plot para gerar o gráfico
        # Aqui a função xi(t) = sum_{j=1}^n Q_{ij}y_j(t) é implementada
        def xi(i, t, n_massas, V, Y0, Lambda):
            soma = 0
            for j in range(0, n_massas):
                soma = soma + Y0[j] * V[i, j] * np.cos(np.sqrt(Lambda[j]) * t)
            return soma

        # Solicitar ao usuário o tempo de simulação desejado
        tempo_de_simulacao=0
        try:
            tempo_de_simulacao = float(input('Digite o tempo de simulação: tempo = '))
        except:
            print('Digitação errada')
            self.__gerarGrafico()

        # Gerar gráfico
        fig, axs = plt.subplots(int(math.ceil(self.n_massas/2)), 2)
        t = np.arange(0, tempo_de_simulacao, tempo_de_simulacao/10000)
        for i in range(0, self.n_massas):
            ax = axs[int(math.floor(i/2)),i % 2]
            ax.plot(t, xi(i, t, self.n_massas, self.V, self.Y0, self.Lambda), label='x{0:d}(t)'.format(i))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, symmetric=False, min_n_ticks=5))
            ax.grid()
            ax.legend(handlelength=-0.4)
            ax.set_xlabel('tempo [s]')
            ax.set_ylabel('posição [m]')

        plt.ioff()
        plt.show()


# Programa principal
# Instancia a classe singleton e chama seu método público
calculadora = Interface()
calculadora.promptTelaInicial()