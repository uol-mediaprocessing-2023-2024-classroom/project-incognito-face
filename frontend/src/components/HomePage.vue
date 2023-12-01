<template>
  <v-container>
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
                <v-list-item v-for="(filter, index) in filters" :key="index">
                  <v-list-item-content @click="selectFilter(filter)">
                    <v-list-item-title>{{ filter.displayName }}</v-list-item-title>
                  </v-list-item-content>
                </v-list-item>
              </v-list>
            </v-menu>
            <button class="basicButton" @click="applyFilter(selectedImage)">
              Apply Filter
            </button>
            <button class="basicButton" @click="runFaceDetection(selectedImage.name)">
              Detect Face
            </button>

            <div>
              <h3>Image Info:<br /></h3>
              <p>
                {{ imageInfo.name }}
              </p>
              <p>
                {{ imageInfo.timestamp }}
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
          <tbody>
            <tr>
              <td class="resultTable">
                <img class="resultImg" v-bind:src="faceResult['viola-jones'].url" />
              </td>
              <td class="resultTable">
                <div>
                  <h3>Viola-Jones<br /></h3>
                  <p>
                    {{ faceResultInfo["viola-jones"].percentage }}
                  </p>
                </div>
              </td>
            </tr>
            <tr>
              <td class="resultTable">
                <img class="resultImg" v-bind:src="faceResult['hog-svn'].url" />
              </td>
              <td class="resultTable">
                <div>
                  <h3>HOG-SVM<br /></h3>
                  <p>
                    {{ faceResultInfo["hog-svn"].percentage }}
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
      <button class="loadMoreBtn" @click="$emit('loadMore')">Load more</button>
    </div>
  </v-container>
</template>

<script>
export default {
  name: "HomePage",

  data() {
    return {
      imageInfo: {
        name: "",
        timestamp: "",
      },
      faceResultInfo: {
        "viola-jones": {
          percentage: "",
        },
        "hog-svn": {
          percentage: "",
        },
      },
      filters: [],
    };
  },

  created() {
    this.loadFilters();
    this.loadImages();
  },

  props: {
    selectedImage: Object,
    selectedFilter: Object,
    currentGallery: Array,
    currentFilters: Array,
    faceResult: Object,
  },

  methods: {
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
    runFaceDetection(image) {
      this.$emit("getFace", image);
    },
  },

  computed: {
    galleryImageNum() {
      return this.currentGallery.length;
    },
  },

  watch: {
    selectedImage: function () {
      const timestamp =
        this.selectedImage.timestamp > 0
          ? new Date(this.selectedImage.timestamp * 1000).toLocaleString("de-DE", {
              hour12: false,
            })
          : "";
      this.imageInfo = {
        name: `Name: ${this.selectedImage.name}`,
        timestamp: `Timestamp: ${timestamp}`,
      };
    },
    currentFilters: function () {
      this.filters = this.currentFilters;
    },
    faceResult: {
      handler: function () {
        Object.keys(this.faceResult).forEach((key) => {
          this.faceResultInfo[key].percentage = this.faceResult[key].percentage + "%";
        });
      },
      deep: true,
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
  max-width: 200px;
  max-height: 200px;
}

.resultTable {
  width: 200px;
}

.basicButton {
  background-color: rgb(226, 215, 215);
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 180px;
}

.basicDropdown {
  background-color: rgb(226, 215, 215) !important;
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

.loadMoreBtn {
  background-color: #a7a7a7;
  border-radius: 6px;
  padding-left: 5px;
  padding-right: 5px;
  width: 100px;
  align-self: center;
  margin-top: 10px;
}
</style>
