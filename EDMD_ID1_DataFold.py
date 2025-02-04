#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:36:10 2025

@author: carolinapessoa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datafold.pcfold import TSCDataFrame
from datafold.appfold import EDMD
from datafold.dynfold.transform import TSCPolynomialFeatures
from datafold.dynfold.dmd import DMDStandard

# Importação e Preparação dos Dados
data = pd.read_csv("penduloduplo_original.csv", sep=";")

# Garantir estrutura correta
expected_columns = ["x_red", "y_red", "x_green", "y_green", "x_blue", "y_blue"]
if not all(col in data.columns for col in expected_columns):
    raise ValueError("As colunas do CSV não correspondem às esperadas!")

# Criando índice de tempo e ID
data["time"] = np.arange(len(data))
data["id"] = 1

# Criando um TSCDataFrame (Time Series Collection DataFrame)
tsc_data = TSCDataFrame(data.set_index(["id", "time"]))

# Aplicação do EDMD com Base Polinomial (grau 2)
dict_step = [("polynomial", TSCPolynomialFeatures(degree=2))]
edmd_poly = EDMD(dict_steps=dict_step, include_id_state=True).fit(tsc_data)

# Obtendo os Autovalores de Koopman
koopman_eigenvalues = np.array(edmd_poly.koopman_eigenvalues)  # Garantir formato correto
print("Autovalores de Koopman:")
print(koopman_eigenvalues[:5])

# Gráfico dos Autovalores de Koopman
plt.figure(figsize=(8, 6))
plt.scatter(np.real(koopman_eigenvalues), np.imag(koopman_eigenvalues), color="blue", label="Autovalores")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label="Círculo Unitário")
plt.gca().add_artist(circle)
plt.title("Autovalores de Koopman no Plano Complexo")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()


# Reconstrução das trajetórias das massas
state_reconstructed = edmd_poly.predict(tsc_data.initial_states(), time_values=tsc_data.time_values())

# Gráficos das Trajetórias Originais vs. Reconstruídas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajetória Original
axes[0].plot(data["x_green"], data["y_green"], label="Massa Verde", color="green")
axes[0].plot(data["x_blue"], data["y_blue"], label="Massa Azul", color="blue")
axes[0].scatter(data["x_red"].iloc[0], data["y_red"].iloc[0], color="red", label="Ponto Fixo", zorder=5)
axes[0].set_title("Trajetória Original")
axes[0].set_xlabel("Posição X")
axes[0].set_ylabel("Posição Y")
axes[0].legend()
axes[0].grid()

# Trajetória Reconstruída
axes[1].plot(state_reconstructed.iloc[:, 2], state_reconstructed.iloc[:, 3], label="Massa Verde", color="green")
axes[1].plot(state_reconstructed.iloc[:, 4], state_reconstructed.iloc[:, 5], label="Massa Azul", color="blue")
axes[1].scatter(state_reconstructed.iloc[0, 0], state_reconstructed.iloc[0, 1], color="red", label="Ponto Fixo", zorder=5)
axes[1].set_title("Trajetória Reconstruída pelo EDMD")
axes[1].set_xlabel("Posição X")
axes[1].set_ylabel("Posição Y")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

from scipy.signal import detrend
from scipy.fftpack import fft, fftfreq

# Plotagem dos Modos Dinâmicos ao Longo do Tempo

# Extraindo os primeiros dois modos dinâmicos reconstruídos
mode_1 = state_reconstructed.iloc[:, 0] - state_reconstructed.iloc[:, 0].mean()
mode_2 = state_reconstructed.iloc[:, 1] - state_reconstructed.iloc[:, 1].mean()

# Removendo tendências lineares dos modos dinâmicos
mode_1_detrended = detrend(mode_1)
mode_2_detrended = detrend(mode_2)

# Criando eixo temporal corretamente baseado no número total de frames
t = np.arange(len(state_reconstructed))

# Modos Dinâmicos sem remoção de tendências lineares
plt.figure(figsize=(12, 5))
plt.plot(t, mode_1, label="Modo 1 (Original)", color="blue")
plt.plot(t, mode_2, label="Modo 2 (Original)", color="orange")
plt.axhline(0, color="black", linestyle="--", alpha=0.6)
plt.title("Modos Dinâmicos ao Longo do Tempo (Antes da Remoção da Tendência)")
plt.xlabel("Frames (Tempo)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Modos Dinâmicos após remoção das tendências
plt.figure(figsize=(12, 5))
plt.plot(t, mode_1_detrended, label="Modo 1 (Sem Tendência)", color="blue")
plt.plot(t, mode_2_detrended, label="Modo 2 (Sem Tendência)", color="orange")
plt.xscale("log")  # Escala logarítmica para destacar os primeiros frames
plt.axhline(0, color="black", linestyle="--", alpha=0.6)
plt.title("Modos Dinâmicos ao Longo do Tempo (Corrigidos)")
plt.xlabel("Frames (Tempo, escala log)")
plt.ylabel("Amplitude Normalizada")
plt.legend()
plt.grid()
plt.show()

# Transformada de Fourier (FFT) e Espectro de Frequências dos Modos Dinâmicos

# Definindo a taxa de amostragem
sampling_rate = 100  # Hz
dt = 1 / sampling_rate  # Passo temporal entre frames

# Aplicação da FFT
freqs = fftfreq(len(mode_1_detrended), d=dt)
fft_mode_1 = np.abs(fft(mode_1_detrended))
fft_mode_2 = np.abs(fft(mode_2_detrended))

# Filtrando frequências positivas
valid_indices = freqs > 0
positive_freqs = freqs[valid_indices]
fft_mode_1 = fft_mode_1[valid_indices]
fft_mode_2 = fft_mode_2[valid_indices]

# Normalização 
fft_mode_1 /= np.max(fft_mode_1)
fft_mode_2 /= np.max(fft_mode_2)

# Plot do Espectro de Frequências dos Modos Dinâmicos
plt.figure(figsize=(12, 5))
plt.plot(positive_freqs, fft_mode_1, label="Modo 1 (FFT)", color="blue")
plt.plot(positive_freqs, fft_mode_2, label="Modo 2 (FFT)", color="orange")
plt.xscale("log")  # Escala logarítmica no eixo X
plt.yscale("log")  # Escala logarítmica no eixo Y
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude Espectral Normalizada")
plt.title("Espectro de Frequências dos Modos Dinâmicos (Escala Log-Log)")
plt.legend()
plt.grid()
plt.show()

# Previsão para os próximos 15.000 Frames
num_steps = 15000
t_future = np.arange(len(tsc_data), len(tsc_data) + num_steps)

# Correção do tamanho
t_future = t_future[:num_steps]

state_predicted = edmd_poly.predict(tsc_data.initial_states(), time_values=t_future)

# Correção do tamanho de state_predicted
state_predicted = state_predicted.iloc[:num_steps, :]

# Plotagem das Previsões - Coordenadas X 

# Massa Vermelha (x_red)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
plt.plot(t_future, state_predicted.iloc[:, 0], label="x_red (Previsto)", linestyle="dashed", color="darkred")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão de x_red para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Massa Verde (x_green)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
plt.plot(t_future, state_predicted.iloc[:, 2], label="x_green (Previsto)", linestyle="dashed", color="darkgreen")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão de x_green para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Massa Azul (x_blue)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
plt.plot(t_future, state_predicted.iloc[:, 4], label="x_blue (Previsto)", linestyle="dashed", color="darkblue")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão de x_blue para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Plotagem das Previsões - Coordenadas Y 

# Massa Vermelha (y_red)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
plt.plot(t_future, state_predicted.iloc[:, 1], label="y_red (Previsto)", linestyle="dashed", color="darkred")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão de y_red para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Massa Verde (y_green)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
plt.plot(t_future, state_predicted.iloc[:, 3], label="y_green (Previsto)", linestyle="dashed", color="darkgreen")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão de y_green para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Massa Azul (y_blue)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
plt.plot(t_future, state_predicted.iloc[:, 5], label="y_blue (Previsto)", linestyle="dashed", color="darkblue")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão de y_blue para os Próximos 15000 Frames")
plt.legend()
plt.grid()
plt.show()

# Previsão Iterativa via Koopman (15.000 frames)
n_states = 6  # Número de variáveis de estado
koopman_modes_reduced = edmd_poly.koopman_modes.iloc[:n_states, :n_states].values
koopman_eigenvalues_diag_reduced = np.diag(koopman_eigenvalues[:n_states])
koopman_modes_inv_reduced = np.linalg.pinv(koopman_modes_reduced)

# Pegando o último estado conhecido
state_initial = tsc_data.iloc[-1, :n_states].values

# Criando a previsão iterativa via Koopman
predicted_iterative = [state_initial]
for i in range(1, num_steps):
    next_state = koopman_modes_reduced @ np.linalg.matrix_power(koopman_eigenvalues_diag_reduced, i) @ koopman_modes_inv_reduced @ state_initial
    predicted_iterative.append(next_state)

# Convertendo para numpy array
predicted_iterative = np.array(predicted_iterative)

# Correção do Tamanho
t_future_iter = t_future[:len(predicted_iterative)]

# Plotagem das Previsões - Coordenadas X 

# Massa Vermelha (x_red)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
plt.plot(t_future_iter, predicted_iterative[:, 0], label="x_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão Iterativa de x_red para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()

# Massa Verde (x_green)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
plt.plot(t_future_iter, predicted_iterative[:, 2], label="x_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão Iterativa de x_green para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()

# Massa Azul (x_blue)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
plt.plot(t_future_iter, predicted_iterative[:, 4], label="x_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada X")
plt.title("Previsão Iterativa de x_blue para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()

# Plotagem das Previsões - Coordenadas Y 

# Massa Vermelha (y_red)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
plt.plot(t_future_iter, predicted_iterative[:, 1], label="y_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão Iterativa de y_red para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()

# Massa Verde (y_green)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
plt.plot(t_future_iter, predicted_iterative[:, 3], label="y_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão Iterativa de y_green para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()

# Massa Azul (y_blue)
plt.figure(figsize=(12, 5))
plt.plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
plt.plot(t_future_iter, predicted_iterative[:, 5], label="y_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
plt.xlabel("Tempo (frames)")
plt.ylabel("Coordenada Y")
plt.title("Previsão Iterativa de y_blue para os Próximos 15000 Frames via Koopman")
plt.legend()
plt.grid()
plt.show()
