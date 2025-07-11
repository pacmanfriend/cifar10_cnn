<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { TrainStatus } from '$lib/api';
  import type { Chart as ChartType } from 'chart.js';

  export let status: TrainStatus | null = null;
  export let history: Array<{ epoch: number; loss: number; accuracy: number }> = [];

  let canvas: HTMLCanvasElement;
  let chart: ChartType | null = null;

  onMount(async () => {
    const { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip } =
      await import('chart.js');
    Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip);

    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Loss',
            data: [],
            borderColor: '#e05c5c',
            backgroundColor: 'transparent',
            yAxisID: 'yLoss',
            tension: 0.3,
            pointRadius: 3
          },
          {
            label: 'Accuracy',
            data: [],
            borderColor: '#4a9ede',
            backgroundColor: 'transparent',
            yAxisID: 'yAcc',
            tension: 0.3,
            pointRadius: 3
          }
        ]
      },
      options: {
        animation: false,
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { title: { display: true, text: 'Epoch' } },
          yLoss: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Loss' }
          },
          yAcc: {
            type: 'linear',
            position: 'right',
            min: 0,
            max: 1,
            title: { display: true, text: 'Accuracy' },
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  });

  onDestroy(() => {
    chart?.destroy();
  });

  $: if (chart && history.length > 0) {
    chart.data.labels = history.map((h) => String(h.epoch + 1));
    chart.data.datasets[0].data = history.map((h) => h.loss);
    chart.data.datasets[1].data = history.map((h) => h.accuracy);
    chart.update();
  }
</script>

<section class="panel">
  <h2>Progress</h2>
  {#if status}
    <div class="metrics">
      <div class="metric"><span>Status</span><strong>{status.status}</strong></div>
      <div class="metric"><span>Epoch</span><strong>{status.epoch + 1}</strong></div>
      <div class="metric"><span>Loss</span><strong>{status.loss?.toFixed(4) ?? 'n/a'}</strong></div>
      <div class="metric">
        <span>Accuracy</span><strong
          >{status.accuracy == null ? 'n/a' : `${(status.accuracy * 100).toFixed(2)}%`}</strong
        >
      </div>
    </div>
    {#if status.error}<p class="error">{status.error}</p>{/if}
  {:else}
    <p>No active job</p>
  {/if}
  <canvas bind:this={canvas} class:hidden={history.length === 0}></canvas>
</section>

<style>
  .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }
  canvas {
    margin-top: 16px;
    max-height: 260px;
  }
  canvas.hidden {
    display: none;
  }
  .error {
    color: #e05c5c;
    margin-top: 8px;
  }
</style>
