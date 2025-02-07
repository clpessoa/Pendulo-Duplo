#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:22:25 2025

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
koopman_eigenvalues = np.array(edmd_poly.koopman_eigenvalues) 
print("Autovalores de Koopman:")
print(koopman_eigenvalues[:5])

# Gráfico dos Autovalores de Koopman
plt.figure(figsize=(8, 6))
plt.scatter(np.real(koopman_eigenvalues), np.imag(koopman_eigenvalues), color="blue", label="Autovalores")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label="Círculo Unitário")
plt.gca().add_artist(circle)
plt.xlabel("Parte Real", fontsize=11, fontname="Arial", color="black")
plt.ylabel("Parte Imaginária", fontsize=11, fontname="Arial", color="black")
plt.legend()
plt.axis('equal')
plt.show()


# Reconstrução das trajetórias das massas
state_reconstructed = edmd_poly.predict(tsc_data.initial_states(), time_values=tsc_data.time_values())

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
plt.xlabel("Frames (Tempo)", fontsize=11, fontname="Arial", color="black")
plt.ylabel("Amplitude", fontsize=11, fontname="Arial", color="black")
plt.legend()
plt.show()

# Modos Dinâmicos após remoção das tendências
plt.figure(figsize=(12, 5))
plt.plot(t, mode_1_detrended, label="Modo 1 (Sem Tendência)", color="blue")
plt.plot(t, mode_2_detrended, label="Modo 2 (Sem Tendência)", color="orange")
plt.xscale("log")  # Escala logarítmica para destacar os primeiros frames
plt.axhline(0, color="black", linestyle="--", alpha=0.6)
plt.xlabel("Frames (Tempo, escala log)", fontsize=11, fontname="Arial", color="black")
plt.ylabel("Amplitude Normalizada", fontsize=11, fontname="Arial", color="black")
plt.legend()
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
plt.xlabel("Frequência (Hz)", fontsize=11, fontname="Arial", color="black")
plt.ylabel("Amplitude Espectral Normalizada", fontsize=11, fontname="Arial", color="black")
plt.legend()
plt.show()

# Gráficos das Trajetórias Originais vs. Reconstruídas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajetória Original
axes[0].plot(data["x_green"], data["y_green"], label="Massa Verde", color="green")
axes[0].plot(data["x_blue"], data["y_blue"], label="Massa Azul", color="blue")
axes[0].scatter(data["x_red"].iloc[0], data["y_red"].iloc[0], color="red", label="Ponto Fixo", zorder=5)
axes[0].set_xlabel("Posição X", fontsize=11, fontname="Arial", color="black")
axes[0].set_ylabel("Posição Y", fontsize=11, fontname="Arial", color="black")
axes[0].legend()

# Trajetória Reconstruída
axes[1].plot(state_reconstructed.iloc[:, 2], state_reconstructed.iloc[:, 3], label="Massa Verde", color="green")
axes[1].plot(state_reconstructed.iloc[:, 4], state_reconstructed.iloc[:, 5], label="Massa Azul", color="blue")
axes[1].scatter(state_reconstructed.iloc[0, 0], state_reconstructed.iloc[0, 1], color="red", label="Ponto Fixo", zorder=5)
axes[1].set_xlabel("Posição X", fontsize=11, fontname="Arial", color="black")
axes[1].set_ylabel("Posição Y", fontsize=11, fontname="Arial", color="black")
axes[1].legend()

plt.tight_layout()
plt.show()

# Previsão para os próximos 15.000 Frames
num_steps = 15000
t_future = np.arange(len(tsc_data), len(tsc_data) + num_steps)

# Correção do tamanho
t_future = t_future[:num_steps]

state_predicted = edmd_poly.predict(tsc_data.initial_states(), time_values=t_future)

# Correção do tamanho de state_predicted
state_predicted = state_predicted.iloc[:num_steps, :]

import matplotlib.pyplot as plt

# Plotagem das Previsões - Coordenadas X 
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (x_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 0], label="x_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future, state_predicted.iloc[:, 0], label="x_red (Previsto)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (x_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 2], label="x_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future, state_predicted.iloc[:, 2], label="x_green (Previsto)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (x_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 4], label="x_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future, state_predicted.iloc[:, 4], label="x_blue (Previsto)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()  
plt.show()

# Plotagem das Previsões - Coordenadas Y 
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (y_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 1], label="y_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future, state_predicted.iloc[:, 1], label="y_red (Previsto)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (y_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 3], label="y_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future, state_predicted.iloc[:, 3], label="y_green (Previsto)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (y_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 5], label="y_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future, state_predicted.iloc[:, 5], label="y_blue (Previsto)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()  
plt.show()

#Previsão Iterativa utilizando 6 modos dinâmicos.

# Definir o número de modos a serem usados na previsão iterativa
n_modes_to_use = min(6, len(koopman_eigenvalues))  # Usando 6 modos 
koopman_modes_reduced = edmd_poly.koopman_modes.iloc[:n_modes_to_use, :n_modes_to_use].values
koopman_eigenvalues_diag_reduced = np.diag(koopman_eigenvalues[:n_modes_to_use])
koopman_modes_inv_reduced = np.linalg.pinv(koopman_modes_reduced)

# Pegando o último estado conhecido
state_initial = tsc_data.iloc[-1, :n_modes_to_use].values

# Criando a previsão iterativa via Koopman
predicted_iterative = [state_initial]
for i in range(1, num_steps):
    next_state = koopman_modes_reduced @ np.linalg.matrix_power(koopman_eigenvalues_diag_reduced, i) @ koopman_modes_inv_reduced @ state_initial
    predicted_iterative.append(next_state)

# Convertendo para numpy array
predicted_iterative = np.array(predicted_iterative)


# Correção do Tamanho
t_future_iter = t_future[:len(predicted_iterative)]

import matplotlib.pyplot as plt

# Plotagem das Previsões Iterativas - Coordenadas X 
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (x_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 0], label="x_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 0], label="x_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (x_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 2], label="x_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 2], label="x_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (x_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 4], label="x_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 4], label="x_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()  
plt.show()

# Plotagem das Previsões Iterativas - Coordenadas Y
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (y_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 1], label="y_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 1], label="y_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (y_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 3], label="y_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 3], label="y_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (y_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 5], label="y_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 5], label="y_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()
plt.show()

#Previsão Iterativa utilizando 15 modos dinâmicos.

# Definir o número de modos a serem usados na previsão iterativa
n_modes_to_use = min(15, len(koopman_eigenvalues))  # Usando 15 modos
koopman_modes_reduced = edmd_poly.koopman_modes.iloc[:n_modes_to_use, :n_modes_to_use].values
koopman_eigenvalues_diag_reduced = np.diag(koopman_eigenvalues[:n_modes_to_use])
koopman_modes_inv_reduced = np.linalg.pinv(koopman_modes_reduced)

# Pegando o último estado conhecido
state_initial = tsc_data.iloc[-1, :n_modes_to_use].values

# Criando a previsão iterativa via Koopman
predicted_iterative = [state_initial]
for i in range(1, num_steps):
    next_state = koopman_modes_reduced @ np.linalg.matrix_power(koopman_eigenvalues_diag_reduced, i) @ koopman_modes_inv_reduced @ state_initial
    predicted_iterative.append(next_state)

# Convertendo para numpy array
predicted_iterative = np.array(predicted_iterative)


# Correção do Tamanho
t_future_iter = t_future[:len(predicted_iterative)]

import matplotlib.pyplot as plt

# Plotagem das Previsões Iterativas - Coordenadas X 
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (x_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 0], label="x_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 0], label="x_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (x_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 2], label="x_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 2], label="x_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (x_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 4], label="x_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 4], label="x_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()  
plt.show()

# Plotagem das Previsões Iterativas - Coordenadas Y
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (y_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 1], label="y_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 1], label="y_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (y_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 3], label="y_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 3], label="y_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (y_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 5], label="y_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 5], label="y_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()
plt.show()

#Previsão Iterativa utilizando 27 modos dinâmicos (máximo de modos).

# Definir o número de modos a serem usados na previsão iterativa
n_modes_to_use = min(27, len(koopman_eigenvalues))  # Usa 30 modos ou o máximo disponível
koopman_modes_reduced = edmd_poly.koopman_modes.iloc[:n_modes_to_use, :n_modes_to_use].values
koopman_eigenvalues_diag_reduced = np.diag(koopman_eigenvalues[:n_modes_to_use])
koopman_modes_inv_reduced = np.linalg.pinv(koopman_modes_reduced)

# Pegando o último estado conhecido
state_initial = tsc_data.iloc[-1, :n_modes_to_use].values

# Criando a previsão iterativa via Koopman
predicted_iterative = [state_initial]
for i in range(1, num_steps):
    next_state = koopman_modes_reduced @ np.linalg.matrix_power(koopman_eigenvalues_diag_reduced, i) @ koopman_modes_inv_reduced @ state_initial
    predicted_iterative.append(next_state)

# Convertendo para numpy array
predicted_iterative = np.array(predicted_iterative)


# Correção do Tamanho
t_future_iter = t_future[:len(predicted_iterative)]

import matplotlib.pyplot as plt

# Plotagem das Previsões Iterativas - Coordenadas X 
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (x_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["x_red"], label="x_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 0], label="x_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 0], label="x_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (x_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["x_green"], label="x_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 2], label="x_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 2], label="x_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (x_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["x_blue"], label="x_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 4], label="x_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 4], label="x_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada X", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()  
plt.show()

# Plotagem das Previsões Iterativas - Coordenadas Y
fig, ax = plt.subplots(3, 1, figsize=(12, 12))  

# Massa Vermelha (y_red)
ax[0].plot(tsc_data.index.get_level_values(1), tsc_data["y_red"], label="y_red (Original)", color="lightcoral")
ax[0].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 1], label="y_red (Reconstruído - EDMD)", color="darkred")
ax[0].plot(t_future_iter, predicted_iterative[:, 1], label="y_red (Iterativo - Koopman)", linestyle="dashed", color="darkred")
ax[0].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[0].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[0].legend()

# Massa Verde (y_green)
ax[1].plot(tsc_data.index.get_level_values(1), tsc_data["y_green"], label="y_green (Original)", color="lightgreen")
ax[1].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 3], label="y_green (Reconstruído - EDMD)", color="darkgreen")
ax[1].plot(t_future_iter, predicted_iterative[:, 3], label="y_green (Iterativo - Koopman)", linestyle="dashed", color="darkgreen")
ax[1].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[1].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[1].legend()

# Massa Azul (y_blue)
ax[2].plot(tsc_data.index.get_level_values(1), tsc_data["y_blue"], label="y_blue (Original)", color="lightskyblue")
ax[2].plot(tsc_data.index.get_level_values(1), state_reconstructed.iloc[:, 5], label="y_blue (Reconstruído - EDMD)", color="darkblue")
ax[2].plot(t_future_iter, predicted_iterative[:, 5], label="y_blue (Iterativo - Koopman)", linestyle="dashed", color="darkblue")
ax[2].set_xlabel("Tempo (frames)", fontsize=11, fontname="Arial", color="black")
ax[2].set_ylabel("Coordenada Y", fontsize=11, fontname="Arial", color="black")
ax[2].legend()

plt.tight_layout()
plt.show()

num_total_modos = len(koopman_eigenvalues)
print(f"Total de modos de Koopman: {num_total_modos}")

num_total_modos = edmd_poly.koopman_modes.shape[1]
print(f"Total de modos de Koopman disponíveis: {num_total_modos}")



