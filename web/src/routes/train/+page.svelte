<script lang="ts">
  import TrainForm from '$lib/components/TrainForm.svelte';
  import TrainProgress from '$lib/components/TrainProgress.svelte';
  import { fetchTrainStatus, startTrain } from '$lib/api';
  import type { TrainRequest, TrainStatus } from '$lib/api';

  let status: TrainStatus | null = null;
  let jobId = '';

  async function start(request: TrainRequest) {
    const job = await startTrain(request);
    jobId = job.job_id;
    status = await fetchTrainStatus(jobId);
  }

  async function refresh() {
    if (jobId) status = await fetchTrainStatus(jobId);
  }
</script>

<div class="grid">
  <TrainForm backend="gpu" onStart={start} />
  <div>
    <TrainProgress {status} />
    <div class="actions refresh">
      <button class="secondary" on:click={refresh}>Refresh</button>
    </div>
  </div>
</div>

<style>
  .refresh {
    margin-top: 12px;
  }
</style>
