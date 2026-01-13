import sys
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

"""
Script para visualizar heatmaps dos agentes
Uso: python visualizar_heatmap.py heatmap_QF1.csv [--environment foraging.json]
"""

def load_heatmap(filename):
    """Carrega dados do heatmap de um arquivo CSV"""
    heatmap_data = {}
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row['x'])
                y = int(row['y'])
                visits = int(row['visits'])
                heatmap_data[(x, y)] = visits
        print(f"✅ Heatmap carregado: {len(heatmap_data)} posições")
        return heatmap_data
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao carregar heatmap: {e}")
        sys.exit(1)

def load_environment(env_json):
    """Carrega informações do ambiente do JSON"""
    try:
        with open(env_json, 'r') as f:
            config = json.load(f)
        return config.get('environment', {})
    except FileNotFoundError:
        print(f"⚠️  Arquivo de ambiente não encontrado: {env_json}")
        return None
    except Exception as e:
        print(f"⚠️  Erro ao carregar ambiente: {e}")
        return None

def visualize_heatmap(heatmap_data, env_config=None, title="Heatmap do Agente"):
    """Visualiza o heatmap com matplotlib"""

    # Determinar dimensões da grelha
    if env_config:
        width = env_config.get('width', 10)
        height = env_config.get('height', 10)
    else:
        # Inferir das posições
        if heatmap_data:
            max_x = max(x for x, y in heatmap_data.keys())
            max_y = max(y for x, y in heatmap_data.keys())
            width = max_x + 1
            height = max_y + 1
        else:
            width, height = 10, 10

    # Criar matriz de calor
    heatmap_matrix = np.zeros((height, width))
    for (x, y), count in heatmap_data.items():
        if 0 <= x < width and 0 <= y < height:
            heatmap_matrix[y, x] = count

    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # Colormap personalizado (branco -> amarelo -> laranja -> vermelho)
    colors = ['white', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Plot do heatmap
    im = ax.imshow(heatmap_matrix, cmap=cmap, interpolation='nearest', origin='upper')

    # Adicionar colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Número de Visitas', rotation=270, labelpad=20)

    # Adicionar grid
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Adicionar labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Adicionar elementos do ambiente se disponível
    if env_config:
        # Ninho
        if 'ninho' in env_config:
            nx, ny = env_config['ninho']
            ax.plot(nx, ny, 'bs', markersize=15, label='Ninho', markeredgecolor='black', markeredgewidth=2)

        # Farol
        if 'farol' in env_config:
            fx, fy = env_config['farol']
            ax.plot(fx, fy, 'g*', markersize=20, label='Farol', markeredgecolor='black', markeredgewidth=2)

        # Paredes
        if 'walls' in env_config:
            walls = env_config['walls']
            for wx, wy in walls:
                rect = plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1,
                                      facecolor='black', alpha=0.8, edgecolor='white')
                ax.add_patch(rect)

        # Recursos (apenas posições iniciais)
        if 'resources' in env_config:
            resources = env_config['resources']
            for rx, ry in resources:
                ax.plot(rx, ry, 'go', markersize=8, alpha=0.5, markeredgecolor='darkgreen')

        # Adicionar legenda
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # Adicionar anotações com contagens nas células mais visitadas
    max_visits = np.max(heatmap_matrix) if heatmap_matrix.size > 0 else 1
    if max_visits > 0:
        threshold = max_visits * 0.3  # Mostrar apenas células com >30% do máximo
        for (x, y), count in heatmap_data.items():
            if count >= threshold and 0 <= x < width and 0 <= y < height:
                ax.text(x, y, str(count), ha='center', va='center',
                       color='white' if count > max_visits * 0.6 else 'black',
                       fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    if len(sys.argv) < 2:
        print("Uso: python visualizar_heatmap.py heatmap_QF1.csv [--environment foraging.json]")
        sys.exit(1)

    heatmap_file = sys.argv[1]
    env_file = None

    # Procurar argumento --environment
    if '--environment' in sys.argv:
        idx = sys.argv.index('--environment')
        if idx + 1 < len(sys.argv):
            env_file = sys.argv[idx + 1]

    print(f"📊 Carregando heatmap de: {heatmap_file}")
    heatmap_data = load_heatmap(heatmap_file)

    total_visits = sum(heatmap_data.values())
    unique_positions = len(heatmap_data)
    print(f"   Total de visitas: {total_visits}")
    print(f"   Posições únicas: {unique_positions}")

    env_config = None
    if env_file:
        print(f"🗺️  Carregando ambiente de: {env_file}")
        env_config = load_environment(env_file)

    # Extrair nome do agente do filename
    agent_name = heatmap_file.replace('heatmap_', '').replace('.csv', '')
    title = f"Mapa de Calor - Agente {agent_name}"

    print("📈 Gerando visualização...")
    fig = visualize_heatmap(heatmap_data, env_config, title)

    # Salvar imagem
    output_file = heatmap_file.replace('.csv', '.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Imagem salva: {output_file}")

    # Mostrar
    plt.show()

if __name__ == '__main__':
    main()

