<script lang="ts">
  import ImageDropzone from '$lib/components/ImageDropzone.svelte';
  import PredictResult from '$lib/components/PredictResult.svelte';
  import { predictImage } from '$lib/api';
  import type { PredictResponse } from '$lib/api';

  let result: PredictResponse | null = null;
  let error = '';

  async function predict(file: File) {
    error = '';
    try {
      result = await predictImage(file);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<div class="grid">
  <ImageDropzone onSelect={predict} />
  <PredictResult {result} />
</div>
{#if error}
  <section class="panel error">{error}</section>
{/if}

<style>
  .error {
    margin-top: 16px;
    color: #b42318;
  }
</style>
