<template>
  <v-container>
    <div id="toggle">
       <Toggle @change_view="change_view" />
    </div>
    <div class="selectedImageField">
      <div class="selectedImageContainer">
        <div class="selectedImageInfo">
          <h2>Selected Image: <br /></h2>
        </div>
        <div style="display: flex">
          <img
            class="selectedImg"
            v-bind:src="selectedImage ? selectedImage.base64 : ''"
          />
          <div class="inputField">
            <button class="basicButton" @click="loadImages()">Reload Images</button>
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
            <button class="basicButton" @click="applyFilter(selectedImage)">
              Apply Filter
            </button>
            <button class="basicButton" @click="handleDetectionButtonClick">
              {{ detectionButtonText }}
            </button>
            <button class="basicButton" @click="downloadImage()">Download Image</button>

            <div>
              <h3>Image Info:<br /></h3>
              <p>
                {{
                  selectedImage && selectedImage.name ? "Name: " + selectedImage.name : ""
                }}
              </p>
              <p>
                {{
                  selectedImage && selectedImage.timestamp > 0
                    ? "Date: " +
                      new Date(selectedImage.timestamp * 1000).toLocaleString("de-DE", {
                        hour12: false,
                      })
                    : ""
                }}
              </p>
              <p>
                {{
                  selectedImage && selectedImage.hash
                    ? "Hash: " + selectedImage.hash.slice(0, 15) + "..."
                    : ""
                }}
              </p>
            </div>
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
                  :src="
                    getFaceImage(algorithm.name)
                      ? getFaceImage(algorithm.name).base64
                      : ''
                  "
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
            <tr v-for="algorithm in currentAlgorithms" :key="algorithm.name">
              <td class="resultTable">
                <img
                  class="resultImg"
                  :src="
                    getFaceImage(algorithm.name)
                      ? getFaceImage(algorithm.name).base64
                      : ''
                  "
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
        </table>
      </div>
    </div>

    <div class="imageGalleryField">
      <div>
        <v-row>
          <v-col v-for="n in galleryImageNum" :key="n" class="d-flex child-flex" cols="2">
            <v-img
              :src="currentGallery[n - 1].base64"
              aspect-ratio="1"
              max-height="200"
              max-width="200"
              class="grey lighten-2"
              @click="selectImage(currentGallery[n - 1])"
            >
              <template v-slot:placeholder>
                <v-row class="fill-height ma-0" align="center" justify="center">
                  <v-progress-circular
                    indeterminate
                    color="grey lighten-5"
                  ></v-progress-circular>
                </v-row>
              </template>
            </v-img>
          </v-col>
        </v-row>
      </div>
      <button class="galleryBtn" @click="$emit('loadMore')">Load more</button>
      <button class="galleryBtn" @click="$refs.fileInput.click()">Upload Image</button>
      <input
        class="fileInput"
        type="file"
        id="imageInput"
        accept=".jpg,.png"
        ref="fileInput"
        @change="uploadFile"
      />
    </div>
  </v-container>
</template>

<script>
import Toggle from './Toggle.vue';
export default {
  name: "HomePage",

  created() {
    this.loadFilters();
    this.loadImages();
  },

   data() {
    return {
      selectedFaceDetection: true
    };
   },

  props: {
    selectedImage: Object,
    selectedFilter: Object,
    currentGallery: Array,
    currentFilters: Array,
    currentAlgorithms: Array,
    faceResult: Array,
    autoDetectionMode: Boolean,
  },

  methods: {
    change_view() {
      this.$emit("change_view");
      this.selectedFaceDetection = !this.selectedFaceDetection
    },
    loadImages() {
      this.$emit("loadImages");
    },
    loadFilters() {
      this.$emit("loadFilters");
    },
    selectImage(image) {
      this.$emit("selectImage", image);
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
        this.$emit("runFaceDetection", this.selectedImage);
      }
    },
    getFaceImage(name) {
      if (this.faceResult == null || this.faceResult.length <= 0) {
        return "";
      }
      return this.faceResult.find((obj) => obj.name === name);
    },
    downloadImage() {
      const base64 = this.selectedImage.base64.split(",")[1];
      const contentType = this.selectedImage.base64
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
      link.download = this.selectedImage.name;
      link.click();
      URL.revokeObjectURL(link.href);
    },
    uploadFile(event) {
      const file = event.target.files[0];
      if (file) {
        this.$emit("uploadImage", file);
      } else {
        alert("No file selected!");
      }
    },
  },
  components: {
    Toggle,
  },
  computed: {
    detectionButtonText() {
      return this.autoDetectionMode ? "Auto Face Detection" : "Run Face Detection";
    },
    galleryImageNum() {
      return this.currentGallery.length;
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

.imageGalleryField {
  display: flex;
  flex-direction: column;
  background-color: rgb(249, 251, 255);
  border-radius: 10px;
  box-shadow: 0 10px 10px 10px rgba(0, 0, 0, 0.1);
  color: black;
  padding: 1%;
  margin-top: 1%;
  max-height: 600px;
  overflow-y: auto;
}

.selectedImg {
  max-width: 500px;
  max-height: 500px;
}

.selectedImageInfo {
  margin-left: 10px;
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

.idInput {
  margin-right: 8px;
  border: 1px solid #000;
  border-radius: 3px;
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

.inputTable {
  width: 80px;
}

.fileInput {
  display: none;
}

.galleryBtn {
  background-color: #a7a7a7;
  border-radius: 6px;
  padding-left: 5px;
  padding-right: 5px;
  width: 150px;
  align-self: center;
  margin-top: 10px;
}

.disabled {
  opacity: 0;
  pointer-events: none;
}

#toggle {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 5px;
}

</style>
