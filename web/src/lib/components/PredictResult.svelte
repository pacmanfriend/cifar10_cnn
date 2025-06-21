<script lang="ts">
  import type { PredictResponse } from '$lib/api';

  export let result: PredictResponse | null = null;
</script>

<section class="panel">
  <h2>Prediction</h2>
  {#if result}
    <div class="metric"><span>Class</span><strong>{result.class_name}</strong></div>
    <div class="metric"><span>Confidence</span><strong>{(result.confidence * 100).toFixed(2)}%</strong></div>
    <div class="scores">
      {#each result.scores as score, index}
        <div><span>{index}</span><progress value={score} max="1"></progress></div>
      {/each}
    </div>
  {:else}
    <p>No prediction</p>
  {/if}
</section>

<style>
  .scores {
    display: grid;
    gap: 8px;
    margin-top: 14px;
  }

  .scores div {
    display: grid;
    grid-template-columns: 28px 1fr;
    align-items: center;
    gap: 10px;
  }

  progress {
    width: 100%;
  }
</style>
