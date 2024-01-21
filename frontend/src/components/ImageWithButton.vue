<template>
  <div class="image-container">
    <div class="selectedImageInfo">
          <h2>{{header}} <br /></h2>
    </div>

    <button v-if="!this.isResult" @click="resetImage()"
            class="resetButton"
          :class="{ 'buttonHide': selectedImage === null || isOriginal}">
      <i style="font-size:24px" class="fa">&#xf021;</i>
    </button>
    <button v-if="!this.isResult" @click="deleteImage()"
            class="deleteButton"
          :class="{ 'buttonHide': selectedImage === null}">
      <i style="font-size:24px" class="fa">&#xf021;</i>
    </button>
    <button v-if="!this.isResult" @click="downloadImage()"
            class="downloadButton"
          :class="{ 'buttonHide': selectedImage === null  || isOriginal}">
      <i style="font-size:24px" class="fa">&#xf021;</i>
    </button>

    <img v-bind:src="selectedImage ? selectedImage.base64 : require('@/assets/placeholder.json').base64" alt="Your Image"
         :class="{ 'resultImg': isResult, 'selectedImgBeforeUpload': selectedImage === null, 'selectedImgAfterUpload': selectedImage !== null }" />
    <button @click="$refs.fileInput.click()"
          :class="{ 'basicButtonBeforeUpload': selectedImage === null, 'buttonHide': selectedImage !== null }"> Upload Image </button>
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
    isResult: Boolean,
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
    deleteImage() {
      this.$emit("deleteImage", this.isOriginal);
    },
    downloadImage() {
      this.$emit("downloadImage", this.selectedImage);
    }
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

.resultImg {
  max-width: 100%;
  max-height: 100%;
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

.buttonHide {
  display: none;
}

.resetButton {
  position: absolute;
  background-color: rgb(204, 203, 205);
  margin-right: 5px;
  border-radius: 3px;
  width: 40px;
  height: 40px;
}

.deleteButton {
  position: absolute;
  background-color: rgb(204, 203, 205);
  right: 0px;
  border-radius: 3px;
  width: 40px;
  height: 40px;
}

.downloadButton {
  position: absolute;
  background-color: rgb(204, 203, 205);
  right: 0px;
  bottom: 7px;
  border-radius: 3px;
  width: 40px;
  height: 40px;
}

.fileInput {
  display: none;
}
</style>