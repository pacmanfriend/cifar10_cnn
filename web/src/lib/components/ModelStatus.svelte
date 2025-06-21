<script lang="ts">
  import type { ModelInfo } from '$lib/api';

  export let model: ModelInfo | null = null;
</script>

<section class="panel">
  <h2>Model</h2>
  {#if model}
    <div class="metric"><span>Backend</span><strong>{model.backend}</strong></div>
    <div class="metric"><span>Loaded</span><strong>{model.loaded ? 'yes' : 'no'}</strong></div>
    <div class="metric"><span>Parameters</span><strong>{model.param_count}</strong></div>
    <p>{model.architecture}</p>
  {:else}
    <p>Unavailable</p>
  {/if}
</section>

<style>
  p {
    margin: 12px 0 0;
    color: #475569;
    line-height: 1.45;
  }
</style>
