<template>
  <v-container>
    <div id="toggle">
       <Toggle @changeView="changeView" />
    </div>
    <div class="selectedImageField">
      <div class="selectedImageContainer">
        <div style="display: flex">
          <div v-if="this.selectedFaceDetection" class="selectedImageDisplay">
            <ImageWithButton class="defaultImage" @uploadImage="uploadImage" @resetImage="resetImage" header="Original Image" :isOriginal=true :selectedImage="originalImage" />
            <ImageWithButton :class="{ 'hideImage': modifiedImage === null, 'defaultImage': modifiedImage !== null }"
                             @uploadImage="uploadImage" @resetImage="resetImage" header="Modified Image" :isOriginal=false :selectedImage="modifiedImage"/>
          </div>
          <div v-else class="selectedImageDisplay">
            <ImageWithButton class="defaultImage" @uploadImage="uploadImage" @resetImage="resetImage" header="Original Image" :isOriginal=true :selectedImage="originalImage" />
            <ImageWithButton class="defaultImage" @uploadImage="uploadImage" @resetImage="resetImage" header="Modified Image" :isOriginal=false :selectedImage="modifiedImage"/>
          </div>
          <div class="inputField">
            <v-menu offset-y>
              <template v-slot:activator="{ on }">
                <v-btn v-on="on" class="basicDropdown">
                  {{ selectedFilter ? selectedFilter.displayName : "Select Filter" }}
                </v-btn>
              </template>
              <v-list>
                <v-list-item
                  v-for="(filter, index) in currentFilters ? currentFilters : []"
                  :key="index"
                >
                  <v-list-item-content @click="selectFilter(filter)">
                    <v-list-item-title>{{ filter.displayName }}</v-list-item-title>
                  </v-list-item-content>
                </v-list-item>
              </v-list>
            </v-menu>
            <button class="basicButton" @click="applyFilter(modifiedImage)">
              Apply Filter
            </button>
            <button v-if="this.selectedFaceDetection" class="basicButton" @click="handleDetectionButtonClick">
              {{ detectionButtonText }}
            </button>
            <button v-else class="basicButton" @click="handleRecognitionButtonClick">Run Face Recognition</button>
            <button class="basicButton" @click="downloadImage()">Download Image</button>
          </div>
        </div>
      </div>
      <div>
        <table>
          <thead>
            <tr>
              <th class="resultTable">
                <div class="resultImgInfo">
                  <h2>Result<br /></h2>
                </div>
              </th>
              <th class="resultTable">
                <div class="resultImgInfo">
                  <h2>Info<br /></h2>
                </div>
              </th>
            </tr>
          </thead>
          <tbody v-if="this.selectedFaceDetection">
            <tr v-for="algorithm in currentAlgorithms" :key="algorithm.name">
              <td class="resultTable">
                <img
                  class="resultImg"
                  :src="getFaceImage(algorithm.name)? getFaceImage(algorithm.name).base64: ''"
                />
              </td>
              <td class="resultTable">
                <div>
                  <h3>{{ algorithm.displayName }}<br /></h3>
                  <p>
                    {{
                      getFaceImage(algorithm.name)
                        ? getFaceImage(algorithm.name).metadata
                        : ""
                    }}
                  </p>
                </div>
              </td>
            </tr>
          </tbody>
          <tbody v-else>
            <tr>
              <td class="resultTable">
                <img
                  class="resultImg"
                  :src="
                    getFaceRecognitionImage()
                      ? getFaceRecognitionImage()[0].base64
                      : ''
                  "
                />
              </td>
              <td class="resultTable">
                <div>
                  <p>
                    {{
                      getFaceRecognitionImage()
                        ? getFaceRecognitionImage()[0].metadata
                        : ""
                    }}
                  </p>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </v-container>
</template>

<script>
import Toggle from './Toggle.vue';
import ImageWithButton from "./ImageWithButton.vue";

export default {
  name: "HomePage",

  created() {
    this.loadFilters();
  },

  props: {
    selectedFaceDetection: Boolean,
    selectedFilter: Object,
    originalImage: Object,
    modifiedImage: Object,
    currentFilters: Array,
    currentAlgorithms: Array,
    autoDetectionMode: Boolean,
    faceResult: Array,
    faceRecognitionResult: Array,
  },

  methods: {
    changeView() {
      this.$emit("changeView");
    },
    uploadImage(file, isOriginal) {
      this.$emit("uploadImage", file, isOriginal, this.selectedFaceDetection);
    },
    resetImage(isOriginal) {
      this.$emit("resetImage", isOriginal, this.selectedFaceDetection);
    },
    loadFilters() {
      this.$emit("loadFilters");
    },
    selectFilter(filter) {
      this.$emit("selectFilter", filter);
    },
    applyFilter(image) {
      this.$emit("applyFilter", image);
    },
    handleDetectionButtonClick(event) {
      if (this.autoDetectionMode || event.shiftKey) {
        this.$emit("toggleAutoDetectionMode");
      } else if (!this.autoDetectionMode) {
        this.$emit("runFaceDetection", this.modifiedImage);
      }
    },
    handleRecognitionButtonClick() {
      this.$emit("runFaceRecognition", this.modifiedImage);
    },
    getFaceImage(name) {
      if (this.faceResult == null || this.faceResult.length <= 0) {
        return "";
      }
      return this.faceResult.find((obj) => obj.name === name);
    },
    getFaceRecognitionImage() {
      if (this.faceRecognitionResult == null || this.faceRecognitionResult.length <= 0) {
        return "";
      }
      return this.faceRecognitionResult;
    },
    downloadImage() {
      const base64 = this.modifiedImage.base64.split(",")[1];
      const contentType = this.modifiedImage.base64
        .split(",")[0]
        .split(":")[1]
        .split(";")[0];
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: contentType });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = this.modifiedImage.name;
      link.click();
      URL.revokeObjectURL(link.href);
    },
  },
  components: {
    ImageWithButton,
    Toggle,
  },
  computed: {
    detectionButtonText() {
      return this.autoDetectionMode ? "Auto Face Detection" : "Run Face Detection";
    },
  },
};
</script>

<style scoped>
.selectedImageField {
  display: flex;
  flex-direction: row;
  background-color: rgb(249, 251, 255);
  border-radius: 10px;
  box-shadow: 0 10px 10px 10px rgba(0, 0, 0, 0.1);
  color: black;
  padding: 1%;
}

.selectedImageDisplay {
  flex-direction: column;
}

.defaultImage {

}

.hideImage {
  display: none;
}

.resultImg {
  max-width: 150px;
  max-height: 150px;
}

.resultTable {
  width: 200px;
}

.basicButton {
  background-color: rgb(204, 203, 205);
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 180px;
}

.basicDropdown {
  background-color: rgb(165, 164, 168) !important;
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 180px;
  height: 24px !important;
  box-shadow: none !important;
}

.inputField {
  display: flex;
  flex-direction: column;
  margin-left: 10px;
  width: 200px;
}

.inputField * {
  margin: 5px 0px 5px 0px;
}

#toggle {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 5px;
}

</style>
