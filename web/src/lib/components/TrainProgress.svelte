<script lang="ts">
  import type { TrainStatus } from '$lib/api';

  export let status: TrainStatus | null = null;
</script>

<section class="panel">
  <h2>Progress</h2>
  {#if status}
    <div class="metric"><span>Status</span><strong>{status.status}</strong></div>
    <div class="metric"><span>Epoch</span><strong>{status.epoch}</strong></div>
    <div class="metric"><span>Loss</span><strong>{status.loss?.toFixed(4) ?? 'n/a'}</strong></div>
    <div class="metric">
      <span>Accuracy</span><strong>{status.accuracy == null ? 'n/a' : `${(status.accuracy * 100).toFixed(2)}%`}</strong>
    </div>
    {#if status.error}<p>{status.error}</p>{/if}
  {:else}
    <p>No active job</p>
  {/if}
</section>
