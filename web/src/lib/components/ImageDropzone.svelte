<script lang="ts">
  export let onSelect: (file: File) => void = () => {};
  let preview: string | null = null;

  function pick(file: File | undefined) {
    if (!file) return;
    preview = URL.createObjectURL(file);
    onSelect(file);
  }
</script>

<section class="panel">
  <h2>Image</h2>
  <label class="drop">
    <input type="file" accept="image/png,image/jpeg" on:change={(event) => pick(event.currentTarget.files?.[0])} />
    {#if preview}
      <img src={preview} alt="" />
    {:else}
      <span>Select PNG or JPEG</span>
    {/if}
  </label>
</section>

<style>
  .drop {
    min-height: 220px;
    border: 1px dashed #94a3b8;
    border-radius: 8px;
    display: grid;
    place-items: center;
    background: #f8fafc;
    cursor: pointer;
    overflow: hidden;
  }

  input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  img {
    width: 100%;
    max-height: 280px;
    object-fit: contain;
  }
</style>
