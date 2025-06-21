<script lang="ts">
  import { onMount } from 'svelte';
  import BackendToggle from '$lib/components/BackendToggle.svelte';
  import FilePathInput from '$lib/components/FilePathInput.svelte';
  import ModelStatus from '$lib/components/ModelStatus.svelte';
  import { fetchModel, loadWeights, saveWeights, setBackend } from '$lib/api';
  import type { Backend, ModelInfo } from '$lib/api';

  let model: ModelInfo | null = null;
  let path = 'weights.bin';
  let message = '';

  onMount(async () => {
    model = await fetchModel();
  });

  async function changeBackend(backend: Backend) {
    model = await setBackend(backend);
  }

  async function run(action: 'load' | 'save') {
    message = '';
    try {
      if (action === 'load') await loadWeights(path);
      else await saveWeights(path);
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<div class="grid">
  <ModelStatus {model} />
  <section class="panel">
    <h2>Backend</h2>
    <BackendToggle backend={model?.backend ?? 'cpu'} onChange={changeBackend} />
  </section>
  <section class="panel">
    <h2>Weights</h2>
    <FilePathInput label="Weights path" bind:value={path} />
    <div class="actions">
      <button class="secondary" on:click={() => run('load')}>Load</button>
      <button class="secondary" on:click={() => run('save')}>Save</button>
    </div>
    {#if message}<p>{message}</p>{/if}
  </section>
</div>
