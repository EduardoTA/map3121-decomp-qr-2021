import numpy as np
import math as math

import matplotlib
#matplotlib.use("qt4agg")
from matplotlib import pyplot as plt

class CalculaAutovalsAutovecs:
    def calcula_c(self, alfa, beta, i):
        if abs(alfa[i]) > abs(beta[i]):
            tau = -beta[i] / alfa[i]
            return 1 / math.sqrt(1 + tau * tau)
        else:
            tau = -alfa[i] / beta[i]
            return tau / math.sqrt(1 + tau * tau)

    def calcula_s(self, alfa, beta, i):
        if abs(alfa[i]) > abs(beta[i]):
            tau = -beta[i] / alfa[i]
            return tau / math.sqrt(1 + tau * tau)
        else:
            tau = -alfa[i] / beta[i]
            return 1 / math.sqrt(1 + tau * tau)

    def calcula_mu(self, alfa, beta, m, deslocamentos):
        # Precisamos obter primeiramente dk.
        dk = (alfa[m - 1] - alfa[m]) / 2
        mu = alfa[m] + dk - np.sign(dk) * math.sqrt(dk ** 2 + beta[m - 1] ** 2)
        if deslocamentos:
            return mu
        else:
            return 0

    def normaliza(self, vector):
        soma = 0
        for i in range(0, len(vector)):
            soma = soma + vector[i] ** 2
        return vector / math.sqrt(soma)

    def AutovalAutovec(self, alfa, beta, epsilon, deslocamentos, V):
        c = np.zeros(alfa.shape[0] - 1)  # Vetor que armazena os valores de c em cada subiteração i da decomposição QR da k-ésima iteração
        s = np.zeros(alfa.shape[0] - 1)  # Vetor que armazena os valores de s em cada subiteração i da decomposição QR da k-ésima iteração
        mu = 0

        # De acordo com o for, esses 3 vetores mudam de função
        Diagonal_principal = np.copy(alfa)
        Subdiagonal_superior = np.copy(beta)
        Subdiagonal_inferior = np.copy(beta)

        # Este vetor é usado para armazenar o valor intermediário de V para que ele seja atribuído no final
        temp_V = np.copy(V)

        # O algoritmo itera sobre matriz m+1 por m+1, que vai diminuindo à medida que os autovalores são encontrados
        m = alfa.shape[0] - 1

        k = 0
        # print('k =', k, ', m =', m, '\nDiagonal principal:', Diagonal_principal, '\nSubdiagonal inferior:',
        #      Subdiagonal_inferior, '\nMatriz V:\n', temp_V, '\n\n')

        while (m >= 0):

            # Estes quatro vetores temp são usados como temporários para armazenar as células da
            # matriz que são atualizadas entre subiterações dos três fors
            temp_diagonal_principal = np.zeros(2)
            temp_subdiagonal = np.zeros(2)

            # Quando multiplicamos o subresultado de V(k+1) por Qi(k), só são modificadas
            # duas colunas do subresultado, aqui estão os temporários das duas colunas modificadas
            temp_V_coluna_esquerda = np.zeros(alfa.shape[0])
            temp_V_coluna_direita = np.zeros(alfa.shape[0])

            # O if só é executado se a submatriz é pelo menos 2x2 e o último beta não puder ser zerado
            if (m > 0 and abs(Subdiagonal_inferior[m - 1]) >= epsilon):
                k = k + 1
                if (k > 1):
                    mu = self.calcula_mu(Diagonal_principal, Subdiagonal_inferior, m, deslocamentos)
                    Diagonal_principal = Diagonal_principal - mu
                # Este for faz a decomposição QR
                #
                # Este for realiza m subiterações i
                # A cada subiteração o subresultado de R(k) é multiplicado por Qi(k)
                # Até no final ter a própria R(k)
                #
                # Funções dos seguinte vetores neste for:
                # Diagonal_principal: Vetor que armazena a diagonal principal do subresultado de R(k), e. no final do for, da própria R
                # Subdiagonal_superior: Vetor que armazena a 1ª subdiagonal acima da diagonal principal do subresultado de R(k), e. no final do for, da própria R(k)
                # Subdiagonal_inferior: Vetor que armazena a 1ª subdiagonal abaixo da diagonal principal do subresultado de R(k), e. no final do for, da própria R(k)
                #
                # Otimizações implementadas neste for:
                # 1. Só são armazenados os cossenos e senos, e não a matriz Qi(k) inteira
                # 2. O elemento da subiteração i do subresultado de R(k) de posição (i,i+1) é zerado automaticamente
                # 3. Só se calcula os valores da diagonal principal e das duas subdiagonais, imediatamente abaixo e acima da principal,
                #    pois os elementos das outras diagonais são zerados e não são usados no cálculo de A(k+1)
                # 4. Entre cada subiteração somente 5 células importantes são alteradas no subresultado de R(k), exceto na última iteração
                for i in range(0, m):
                    # Calcula c e s da subiteração i da decomposição QR #Otimização 1
                    c[i] = self.calcula_c(Diagonal_principal, Subdiagonal_inferior, i)
                    s[i] = self.calcula_s(Diagonal_principal, Subdiagonal_inferior, i)

                    # Calcula os novos valores das duas células atualizadas da diagonal principal do subresultado de R(k)
                    temp_diagonal_principal[0] = c[i] * Diagonal_principal[i] - s[i] * Subdiagonal_inferior[i]
                    temp_diagonal_principal[1] = s[i] * Subdiagonal_superior[i] + c[i] * Diagonal_principal[i + 1]

                    # Calcula os novos valores das duas células atualizadas da 1ª subdiagonal acima da diagonal principal do subresultado de R(k)
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
                # Diagonal_principal: Vetor que armazena a diagonal principal do subresultado de A(k+1), e. no final do for, da própria A(k+1)
                # Subdiagonal_superior: Vetor que armazena a 1ª subdiagonal acima da diagonal principal do subresultado de A(k+1), e. no final do for, da própria A(k+1)
                # Subdiagonal_inferior: Vetor que armazena a 1ª subdiagonal abaixo da diagonal principal do subresultado de A(k+1), e. no final do for, da própria A(k+1)
                #
                # Otimizações implementadas neste for:
                # 1. Como a matriz A(k+1) termina simétrica, então só calculamos o valor para a diagonal imediatamente abaixo da principal
                # 2. Como as multiplicações por Qi(k)T só modificam duas colunas do subresultado de A(k+1), então só é necessário calcular 3 células do subresultado
                for i in range(0, m):
                    # Calcula os novos valores das duas células atualizadas da diagonal principal do subresultado de A(k+1)
                    temp_diagonal_principal[0] = c[i] * Diagonal_principal[i] - s[i] * Subdiagonal_superior[i]
                    temp_diagonal_principal[1] = c[i] * Diagonal_principal[i + 1]

                    # Calcula o novo valos da célula atualizada da 1ª subdiagonal abaixo da diagonal principal do subresultado de A(k+1)
                    temp_subdiagonal[0] = -s[i] * Diagonal_principal[i + 1]  # Otimização 1

                    # Atualiza o subresultado com os valores temporários
                    Diagonal_principal[i] = np.copy(temp_diagonal_principal[0])
                    Diagonal_principal[i + 1] = np.copy(temp_diagonal_principal[1])
                    Subdiagonal_inferior[i] = np.copy(temp_subdiagonal[0])
                    Subdiagonal_superior[i] = np.copy(temp_subdiagonal[
                                                          0])  # Como A(k+1) é simétrica, não precisamos calcular a subdiagonal superior #Otimização 1
                # Este for faz a multiplicação V(k+1)=V(k)*Q(k)T, e armazena o resultado em temp_V
                #
                # Este for realiza m subiterações subiteracao
                # A cada subiteração o subresultado de V(k+1) é multiplicado por Qi(k)T
                # Até no final ter a própria V(k+1)
                #
                # Otimizações implementadas neste for:
                # 1. Como as multiplicações por Qi(k)T só modificam duas colunas do subresultado de V(k+1), então só é necessário calcular 2 colunas do subresultado
                for subiteracao in range(0, m):
                    for linha in range(0, alfa.shape[0]):
                        temp_V_coluna_esquerda[linha] = temp_V[linha, subiteracao] * c[subiteracao] - temp_V[linha, subiteracao + 1] * s[subiteracao]
                        temp_V_coluna_direita[linha] = temp_V[linha, subiteracao + 1] * c[subiteracao] + temp_V[linha, subiteracao] * s[subiteracao]

                        temp_V[linha, subiteracao] = temp_V_coluna_esquerda[linha]
                        temp_V[linha, subiteracao + 1] = temp_V_coluna_direita[linha]

                if (k > 1):
                    Diagonal_principal = Diagonal_principal + mu

            # print('k =', k, ', m =', m, '\nDiagonal principal:', Diagonal_principal, '\nSubdiagonal inferior:',
            #      Subdiagonal_inferior, '\nMatriz V:\n', temp_V, '\n\n')

            # Este if faz a eliminação de beta se este for menor que epsilon e diminui o escopo (m=m-1)
            # do algoritmo, para que ele passe a trabalhar com a submatriz
            if (abs(Subdiagonal_inferior[m - 1]) < epsilon):
                Subdiagonal_inferior[m - 1] = 0.0
                Subdiagonal_superior[m - 1] = 0.0
                m = m - 1

            # Normaliza autovetores
            for j in range(0, alfa.shape[0]):
                temp_V[:, j] = self.normaliza(temp_V[:, j])
        return (Diagonal_principal, temp_V, k)

class EstimadorDeErros:
    def normaliza(self, vector):
        soma = 0
        for i in range(0, len(vector)):
            soma = soma + vector[i] ** 2
        return vector / math.sqrt(soma)

    def EstimativasErroAnalitico(self, n, Lambda, V):
        autovalores_reais = 2 * (1 - np.cos(np.arange(n, 0, -1) * math.pi / (n + 1)))
        autovalores_reais = np.flip(np.sort(autovalores_reais))

        autovalores_obtidos = np.copy(np.flip(np.sort(Lambda)))

        erros = np.zeros(n)

        print('Autovalor obtido | Autovalor real | erro')
        for i in range(0, n):
            erros[i] = math.sqrt(math.pow(autovalores_obtidos[i] - autovalores_reais[i], 2))
            print('{0:12.10f}       {1:12.10f}     {2}'.format(autovalores_obtidos[i], autovalores_reais[i],erros[i]))
        print('-----------------------------')
        print('Erro máx: ', np.max(erros))
        print('\n')

        autovetores_reais = np.zeros((n, n))
        for j in range(0, n):
            autovetores_reais[:, j] = np.sin(np.arange(1, n + 1, 1) * (n - j) * math.pi / (n + 1))
            autovetores_reais[:, j] = self.normaliza(autovetores_reais[:, j])
        print('Erro obtido fazendo max(abs(Autovetores_reais-Autovetores_obtidos)) = {0}\n\n'.format(np.max(abs(autovetores_reais - V))))

    def EstimativasErroGerais(self, n, alfa, beta, Lambda, V):
        erro = np.max(np.abs(np.matmul(V, np.transpose(V)) - np.identity(n)))
        print('max(abs(matmul(Q,QT)-I)) = {0}'.format(erro))

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
        print('max(abs(matmul(A,Q)-matmul(Q,Lambda))) = {0}\n'.format(erro))

        erros = np.zeros(n)
        for i in range(0, n):
            erros[i] = math.sqrt(math.pow(np.mean(np.divide(np.matmul(A, V[:, i]), V[:, i])) - Lambda[i], 2))
            print('Autovalor obtido fazendo matdiv(matmul(A,v),v): {0:12.10f},  Autovalor obtido pelo método QR: {1:12.10f}, erro: {2}'.format(np.mean(np.divide(np.matmul(A, V[:, i]), V[:, i])), Lambda[i], erros[i]))
        print('Erro máximo = {0}\n'.format(np.max(erros)))

    def estimarErro(self, n, alfa, beta, Lambda, V, mostrar_erro_caso_analitico):
        if(mostrar_erro_caso_analitico):
            self.EstimativasErroAnalitico(n, Lambda, V)
        else:
            self.EstimativasErroGerais(n, alfa, beta, Lambda, V)

class Interface:
    alfa = np.zeros(1)
    beta = np.zeros(1)

    n = 0

    usar_caso_analítico = False
    usar_deslocamentos_espectrais = False
    epsilon = 0.000001
    V = np.identity(alfa.shape[0])
    Lambda = 0

    calculadora = CalculaAutovalsAutovecs()
    estimadorDeErros = EstimadorDeErros()

    resultados = ()

    n_massas = 0
    massa = 0.0
    X0 = np.zeros(n_massas)
    Y0 = np.zeros(n_massas)
    ki = np.zeros(n_massas+1)

    def promptTelaInicial(self):
        # Menu de seleção inicial
        print('Selecione uma opção')
        print('(0) para calcular autovalores e autovetores de matriz tridiagonal')
        print('(1) para obter resposta de sistema massa-mola')
        print('(qualquer outra) para sair')
        selecao = input('=')

        if(selecao == '0'):
            self.promptCalcularAutovalsAutovecs()
            print('->Saindo...')
            self.promptTelaInicial()
        elif(selecao == '1'):
            self.promptMassaMola()
            print('->Saindo...')
            self.promptTelaInicial()
        else:
            print('->Saindo...')
            return

    def promptCalcularAutovalsAutovecs(self):
        print('->Selecionado modo de cálculo de autovalores e autovetores') #Confirmação de seleção

        try:
            self.n = int(input('Tamanho da matriz A: n = '))
            print('Resettando matriz V')
            self.V = np.identity(self.n)
            if(input('Usar caso analítico? (y/n)') == 'y'):
                self.usar_caso_analítico = True
                self.alfa = 2*np.ones(self.n)
                self.beta = -1*np.ones(self.n-1)
            else:
                print('Digitar os vetores da seguinte forma (n=3): 1 2 3')
                self.alfa = np.array(list(map(float, input("alfa = ").strip().split()))[:self.n])
                self.beta = np.array(list(map(float, input("beta = ").strip().split()))[:self.n - 1])

            # Solicita ao usuário se deseja usar deslocamentos espectrais
            if(input('Usar deslocamentos espectrais? (y/n) ') == 'y'):
                self.usar_deslocamentos_espectrais = True
            else:
                self.usar_deslocamentos_espectrais = False

            self.epsilon = float(input('epsilon: ep = '))

            if(self.alfa.shape[0] != self.n and self.beta.shape[0] != self.n-1):
                raise Exception()
        except:
            print('Erro de digitação')
            return

        self.promptTelaResultadosCalculoAutovalsAutovecs()

    def promptTelaResultadosCalculoAutovalsAutovecs(self):
        # Aplica o método de obtenção de autovalores e autovetores usando decomposição QR
        self.resultados = self.calculadora.AutovalAutovec(self.alfa, self.beta, self.epsilon, self.usar_deslocamentos_espectrais, self.V)
        self.Lambda = self.resultados[0]
        self.V = self.resultados[1]
        k = self.resultados[2]

        # Número de iterações necessárias
        print('Foram necessárias {0:d} iterações'.format(k))

        # Menu de seleção
        print('Selecione uma opção')
        print('(0) Obter autovalores e autovetores')
        print('(1) Cálculo de estimativas de erro')
        print('(qualquer outra) para sair')
        selecao = input('=')

        if(selecao == '0'):
            self.mostrarAutovalsAutovecs()
            self.promptTelaResultadosCalculoAutovalsAutovecs()
        elif(selecao == '1'):
            self.promptEstimativasDeErro()
            self.promptTelaResultadosCalculoAutovalsAutovecs()
        else:
            return

    def mostrarAutovalsAutovecs(self):
        for i in range(0, self.alfa.shape[0]):
            np.set_printoptions(formatter={'float': '{: 12.10f}'.format})
            print('Autovalor: {0:12.10f}, Autovetor:'.format(self.Lambda[i]), self.V[:, i])

    def promptEstimativasDeErro(self):
        # Porém se for detectada matriz com solução analítica, outro menu surge
        if (np.array_equal(self.alfa, 2 * np.ones(self.n)) and np.array_equal(self.beta, -1 * np.ones(self.n - 1))):
            print('Selecione uma opção')
            print('(0) Estimativas de erro gerais')
            print('(1) Estimativas de erro comparando com o resultado analítico')
            print('(qualquer outra) para sair')
            selecao = input('=')

            if(selecao == '0'):
                self.estimadorDeErros.estimarErro(self.n, self.alfa, self.beta, self.Lambda, self.V, False)
                self.promptEstimativasDeErro()
            elif(selecao == '1'):
                self.estimadorDeErros.estimarErro(self.n, self.alfa, self.beta, self.Lambda, self.V, True)
                self.promptEstimativasDeErro()
            else:
                return
    def promptMassaMola(self):
        print('->Selecionado modo de obtenção de resposta de sistema massa-mola')

        try:
            self.n_massas = int(input('Número de massas: n_massas = '))

            self.V = np.identity(self.n_massas)
            self.X0 = np.zeros(self.n_massas)
            self.Y0 = np.zeros(self.n_massas)
            self.ki = np.zeros(self.n_massas+1)

            self.massa = float(input('Massa = '))

            # Solicita ao usuário se deseja usar deslocamentos espectrais
            if (input('Usar deslocamentos espectrais? (y/n) ') == 'y'):
                self.usar_deslocamentos_espectrais = True
            else:
                self.usar_deslocamentos_espectrais = False
        except:
            print('Digitação errada')
            return
        self.promptMenuMassaMola()

    def promptMenuMassaMola(self):
        print('Selecione qual opção de constante de mola será usada:')
        print('(0): ki = (40+2i)')
        print('(1): ki = (40+2(-1)^i)')
        print('(2): Digitar as condições iniciais do problema (X0)')
        print('(3): Digitar os valores de constantes de mola')
        print('(4): GERAR RESULTADO')
        print('(qualquer outra) para sair')
        selecao = (input('='))

        if(selecao == '0'):
            print('Seleção 0')
            self.ki = 40 + 2 * np.arange(1, self.n_massas + 2)
            self.promptMenuMassaMola()
        elif(selecao == '1'):
            print('Seleção 1')
            self.ki = 40 + 2 * (-1) ** np.arange(1, self.n_massas + 2)
            self.promptMenuMassaMola()
        elif(selecao == '2'):
            print('Seleção 2')
            print('Digitar as condições iniciais (X0):')
            print('Digitar o vetor da seguinte forma: 1 2 3 4 5')

            try:
                self.X0 = np.array(list(map(float, input("X0 = ").strip().split()))[:self.n_massas])
                if (self.X0.shape[0] != self.n_massas):
                    raise Exception()
            except:
                print('Digitação errada')
            self.promptMenuMassaMola()
        elif(selecao == '3'):
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
            self.promptMenuMassaMola()
        elif(selecao == '4'):
            self.alfa = np.ones(self.n_massas)
            self.beta = np.ones(self.n_massas - 1)
            self.V = np.identity(self.n_massas)

            for i in range(0, self.n_massas):
                self.alfa[i] = (self.ki[i] + self.ki[i + 1]) / self.massa
            for i in range(0, self.n_massas - 1):
                self.beta[i] = -self.ki[i + 1] / self.massa

            self.resultados = self.calculadora.AutovalAutovec(self.alfa, self.beta, self.epsilon,
                                                              self.usar_deslocamentos_espectrais, self.V)
            self.Lambda = self.resultados[0]
            self.V = self.resultados[1]

            print(self.resultados[0])

            self.promptTelaResultadosMassaMola()
            self.promptMenuMassaMola()
        else:
            return
    def promptTelaResultadosMassaMola(self):
        print('Selecione qual opção de constante de mola será usada:')
        print('(0): Gerar gráfico')
        print('(1): Imprimir frequências e modos naturais')
        selecao = (input('='))

        if(selecao == '0'):
            self.gerarGrafico()
            self.promptTelaResultadosMassaMola()
        elif(selecao == '1'):
            for i in range(0, self.n_massas):
                np.set_printoptions(formatter={'float': '{: 12.10f}'.format})
                print('freq: {0:12.10f}, Modo natural:'.format(math.sqrt(self.Lambda[i])), self.V[:,i])
            self.promptTelaResultadosMassaMola()
        else:
            print('->Saindo...')
            return
    def gerarGrafico(self):
        for i in range(0, self.n_massas):
            soma = 0
            for j in range(0, self.n_massas):
                soma = soma + np.transpose(self.V)[i, j] * self.X0[j]
            self.Y0[i] = soma

        def xi(i, t, n_massas, V, Y0, Lambda):
            soma = 0
            for j in range(0, n_massas):
                soma = soma + Y0[j] * V[i, j] * np.cos(np.sqrt(Lambda[j]) * t)
            return soma


        try:
            tempo = float(input('Digite o tempo de simulação: tempo = '))
        except:
            print('Digitação errada')
            self.gerarGrafico()

        plt.ion()
        plt.figure(1)

        t = np.arange(0, tempo, tempo/1000)
        for i in range(0, self.n_massas):
            plt.plot(t, xi(i, t, self.n_massas, self.V, self.Y0, self.Lambda), label='x{0:d}(t)'.format(i))
        plt.legend(handlelength=-0.4)
        plt.xlabel('tempo [s]')
        plt.ylabel('posição [m]')
        plt.title('{0}'.format(0))

        plt.ioff()
        plt.show()



calculadora = Interface()
calculadora.promptTelaInicial()