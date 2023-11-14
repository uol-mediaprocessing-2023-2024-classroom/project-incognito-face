<template>
    <v-container>
        <div class="selectedImageField">
            <div class="selectedImageContainer">
                <div class="selectedImageInfo">
                    <h2>Selected Image: <br /></h2>
                </div>

                <div style="display: flex">
                    <img class="selectedImg" v-bind:src="selectedImage.url" />
                    <div class="inputField">
                        <button class="basicButton" @click="loadImages()">
                            Load Images
                        </button>

                        <button class="basicButton" @click="getBlur(selectedImage.name)">
                            Apply Blur
                        </button>

                        <button class="basicButton" @click="getFace(selectedImage.name)">
                            Show Face
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
        </div>

        <div class="imageGalleryField">
            <div>
                <v-row>
                    <v-col v-for="n in galleryImageNum" :key="n" class="d-flex child-flex" cols="2">
                        <v-img :src="currentGallery[n - 1].url" aspect-ratio="1" max-height="200" max-width="200" class="grey lighten-2" @click="updateSelected(currentGallery[n - 1].name)">
                            <template v-slot:placeholder>
                                <v-row class="fill-height ma-0" align="center" justify="center">
                                    <v-progress-circular indeterminate color="grey lighten-5"></v-progress-circular>
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
                timestamp: ""
            },
        };
    },

    props: {
        selectedImage: Object,
        currentGallery: Array,
    },

    methods: {
        // --- IMAGE RELATED METHODS ---

        // Emit a loadImages event.
        loadImages() {
            this.$emit("loadImages");
        },

        // Emit a updateSelected event with the ID of the selected image.
        // This method is called when the user clicks/selects an image in the gallery of loaded images.
        updateSelected(imgName) {
            this.$emit("updateSelected", imgName);
        },

        // Emit a getBlur event with the ID of the selected image.
        getBlur(imgName) {
            console.log(imgName);
            this.$emit("getBlur", imgName);
        },
        getFace(imgName) {
            this.$emit("getFace", imgName);
        },
    },

    computed: {
        /*
            The numer of images within currentGallery can dynamically change after the DOM is loaded. Since the size of the image gallery depends on it,
            it's important for it to be updated within the DOM aswell. By using computed values this is not a problem since Vue updates the DOM in accordance wit them.
        */
        galleryImageNum() {
            return this.currentGallery.length;
        },
    },

    watch: {

        // Watcher function for updating the displayed image information.
        selectedImage: function () {
            const timestamp = this.selectedImage.timestamp > 0 ? new Date(this.selectedImage.timestamp * 1000).toLocaleString('de-DE', { hour12: false }) : '';
            this.imageInfo = { name: `Name: ${this.selectedImage.name}`, timestamp: `Timestamp: ${timestamp}` };
        },
    }
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

.basicButton {
    background-color: rgb(226, 215, 215);
    padding: 0px 4px 0px 4px;
    margin-right: 5px;
    border-radius: 3px;
    width: 150px;
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
    width: 400px;
}

.inputField * {
    margin: 5px 0px 5px 0px;
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
