<template>
  <div class="image-container">
    <div class="selectedImageInfo">
          <h2>{{header}} <br /></h2>
    </div>
    <button @click="resetImage()"
          :class="{ 'resetButtonBeforeUpload': selectedImage === null, 'resetButtonAfterUpload': selectedImage !== null }">
      <i style="font-size:24px" class="fa">&#xf021;</i>
      </button>

    <img v-bind:src="selectedImage ? selectedImage.base64 : require('@/assets/placeholder.json').base64" alt="Your Image"
         :class="{ 'selectedImgBeforeUpload': selectedImage === null, 'selectedImgAfterUpload': selectedImage !== null }" />
    <button @click="$refs.fileInput.click()"
          :class="{ 'basicButtonBeforeUpload': selectedImage === null, 'basicButtonAfterUpload': selectedImage !== null }"> Upload Image </button>
    <input
        class="fileInput"
        type="file"
        id="imageInput"
        accept=".jpg,.png"
        ref="fileInput"
        @click="resetUploadFile"
        @input="uploadFile"
      />

  </div>
</template>

<script>
export default {
  props: {
    isOriginal: Boolean,
    header: String,
    selectedImage: Object,
  },
  methods: {
    resetUploadFile() {
      this.$refs.fileInput.value = '';
    },
    uploadFile(event) {
      const file = event.target.files[0];
      if (file) {
        this.$emit("uploadImage", file, this.isOriginal);
      } else {
        alert("No file selected!");
      }
    },
    resetImage() {
      this.$emit("resetImage", this.isOriginal);
    },
  },
};
</script>

<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css');

.selectedImageInfo {
  margin-left: 10px;
}

.image-container {
  position: relative;
  display: inline-block;
}

.selectedImgBeforeUpload {
  max-width: 500px;
  max-height: 500px;
  opacity: 50%;
}

.selectedImgAfterUpload {
  max-width: 500px;
  max-height: 500px;
  opacity: 100%;
}

.basicButtonBeforeUpload {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  //background-color: rgb(204, 203, 205);
  background-color: green;
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 12px;
  width: 140px;
  height: 60px
}

.basicButtonAfterUpload {
  display: none;
}

.resetButtonBeforeUpload {
  display: none;
}

.resetButtonAfterUpload {
  position: absolute;
  background-color: rgb(204, 203, 205);
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 40px;
  height: 40px;
}

.fileInput {
  display: none;
}
</style>