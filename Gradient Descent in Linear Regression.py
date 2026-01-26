{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPnX0tWb8KkcBqxVFY5WGqK",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/waghvaishnav/Machine-Learning-Algorithms-and-Data-Processing-Playground-hub/blob/main/Gradient%20Descent%20in%20Linear%20Regression.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "# Sample dataset\n",
        "x = np.array([1, 2, 3, 4, 5], dtype=float)\n",
        "y = np.array([2, 4, 6, 8, 10], dtype=float)\n",
        "\n",
        "# Initialize parameters\n",
        "m = 0.0      # slope\n",
        "b = 0.0      # intercept\n",
        "lr = 0.01    # learning rate\n",
        "epochs = 1000\n",
        "n = len(x)\n",
        "\n",
        "# Gradient Descent\n",
        "for i in range(epochs):\n",
        "    y_pred = m * x + b\n",
        "\n",
        "    # Compute gradients\n",
        "    dm = (-2/n) * np.sum(x * (y - y_pred))\n",
        "    db = (-2/n) * np.sum(y - y_pred)\n",
        "\n",
        "    # Update parameters\n",
        "    m -= lr * dm\n",
        "    b -= lr * db\n",
        "\n",
        "    # Print loss every 100 iterations\n",
        "    if i % 100 == 0:\n",
        "        loss = np.mean((y - y_pred) ** 2)\n",
        "        print(f\"Epoch {i}, Loss: {loss:.4f}\")\n",
        "\n",
        "print(\"Final m:\", m)\n",
        "print(\"Final b:\", b)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xly2gFdRCyzu",
        "outputId": "eff3688f-39b2-43b9-8642-36c044316c83"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 0, Loss: 44.0000\n",
            "Epoch 100, Loss: 0.0245\n",
            "Epoch 200, Loss: 0.0124\n",
            "Epoch 300, Loss: 0.0063\n",
            "Epoch 400, Loss: 0.0032\n",
            "Epoch 500, Loss: 0.0016\n",
            "Epoch 600, Loss: 0.0008\n",
            "Epoch 700, Loss: 0.0004\n",
            "Epoch 800, Loss: 0.0002\n",
            "Epoch 900, Loss: 0.0001\n",
            "Final m: 1.9951803506719779\n",
            "Final b: 0.017400463340610635\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# X with multiple features\n",
        "X = np.array([[1, 2],\n",
        "              [2, 3],\n",
        "              [3, 4],\n",
        "              [4, 5]])\n",
        "\n",
        "y = np.array([5, 7, 9, 11])\n",
        "\n",
        "weights = np.zeros(X.shape[1])\n",
        "bias = 0\n",
        "lr = 0.01\n",
        "epochs = 1000\n",
        "n = len(X)\n",
        "\n",
        "for _ in range(epochs):\n",
        "    y_pred = np.dot(X, weights) + bias\n",
        "\n",
        "    dw = (-2/n) * np.dot(X.T, (y - y_pred))\n",
        "    db = (-2/n) * np.sum(y - y_pred)\n",
        "\n",
        "    weights -= lr * dw\n",
        "    bias -= lr * db\n",
        "\n",
        "print(\"Weights:\", weights)\n",
        "print(\"Bias:\", bias)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l2cbn-QtM8yT",
        "outputId": "4cf5c4af-4e95-40f7-9df1-f51c186beedc"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Weights: [0.35583264 1.65797617]\n",
            "Bias: 1.3021435271709496\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "6HMOCrmgM8tQ"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}