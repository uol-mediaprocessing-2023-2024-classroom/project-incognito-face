<template>
  <div>
    <button
      @click="toggleColor(1)"
      :class="{ 'pressed': isPressed === 1, 'default': isPressed !== 1 }"
    >
      Face Detection
    </button>
    <button
      @click="toggleColor(2)"
      :class="{ 'pressed': isPressed === 2, 'default': isPressed !== 2 }"
    >
      Face Recognition
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isPressed: 1,
    };
  },
  methods: {
    toggleColor(buttonNumber) {
      if (this.isPressed !== buttonNumber)
        this.$emit("changeView");
      this.isPressed = buttonNumber;
    },
  },
};
</script>

<style scoped>
.default {
  background-color: rgb(204, 203, 205);
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 180px;
}

.pressed {
  background-color: green;
  padding: 0px 4px 0px 4px;
  margin-right: 5px;
  border-radius: 3px;
  width: 180px;
}
</style>