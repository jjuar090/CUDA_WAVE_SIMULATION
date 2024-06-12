#include <SFML/Graphics.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono> // For time measurement
#include <thread> // For sleeping
#include "CUDA_HEADER_CUH.cuh"

constexpr int gridSizeX = 100;
constexpr int gridSizeY = 100;
constexpr int cellSize = 5;
constexpr float dt = 0.1f;  // Time step
constexpr float damping = 0.99f;  // Damping factor to reduce wave intensity over time

std::vector<float> current(gridSizeX* gridSizeY, 0.0f);
std::vector<float> previous(gridSizeX* gridSizeY, 0.0f);
std::vector<float> temp(gridSizeX* gridSizeY, 0.0f);
std::vector<bool> barriers(gridSizeX* gridSizeY, false);  // Barriers grid

std::chrono::time_point<std::chrono::steady_clock> lastWaveTime;

void applyWave(int mouseX, int mouseY);
void handleEvents(sf::RenderWindow& window);
void update();
void render(sf::RenderWindow& window);
void setBoundary(std::vector<float>& field);
void addBarrier(int x, int y, int width, int height);
void clearMemory();

int main() {
    sf::RenderWindow window(sf::VideoMode(gridSizeX * cellSize, gridSizeY * cellSize), "Wave Simulation");

    // Add some barriers for testing
    addBarrier(20, 20, 10, 10);
    addBarrier(50, 50, 5, 15);

    while (window.isOpen()) {
        handleEvents(window);
        update();
        render(window);
    }

    return 0;
}

void applyWave(int mouseX, int mouseY) {
    int waveSize = 3;  // Smaller wave size for more localized interaction
    float waveStrength = 200.0f;  // Reduced wave strength

    for (int y = mouseY - waveSize; y <= mouseY + waveSize; ++y) {
        for (int x = mouseX - waveSize; x <= mouseX + waveSize; ++x) {
            if (x >= 0 && x < gridSizeX && y >= 0 && y < gridSizeY) {
                int index = y * gridSizeX + x;
                if (!barriers[index]) {
                    current[index] += waveStrength;
                }
            }
        }
    }
}

void handleEvents(sf::RenderWindow& window) {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }
        else if (event.type == sf::Event::MouseButtonPressed) {
            if (event.mouseButton.button == sf::Mouse::Left) {
                int mouseX = event.mouseButton.x / cellSize;
                int mouseY = event.mouseButton.y / cellSize;
                applyWave(mouseX, mouseY);
                lastWaveTime = std::chrono::steady_clock::now(); // Record the time of the wave
            }
        }
    }
}

void update() {
    for (int y = 1; y < gridSizeY - 1; ++y) {
        for (int x = 1; x < gridSizeX - 1; ++x) {
            int index = y * gridSizeX + x;
            if (!barriers[index]) {
                current[index] = ((previous[(y - 1) * gridSizeX + x] +
                    previous[(y + 1) * gridSizeX + x] +
                    previous[y * gridSizeX + x - 1] +
                    previous[y * gridSizeX + x + 1]) / 2 - current[index]) * damping;
            }
        }
    }

    setBoundary(current);

    // Swap buffers
    std::swap(current, previous);

    // Check if it's time to clear memory

}

void render(sf::RenderWindow& window) {
    window.clear(sf::Color::Black);

    sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
    for (int y = 0; y < gridSizeY; ++y) {
        for (int x = 0; x < gridSizeX; ++x) {
            int index = y * gridSizeX + x;
            if (barriers[index]) {
                cell.setFillColor(sf::Color::White);  // Barriers are white
            }
            else {
                float value = current[index];
                sf::Color color = value > 0 ? sf::Color(0, 0, std::min(static_cast<int>(std::abs(value) * 255.0f), 255))
                    : sf::Color(std::min(static_cast<int>(std::abs(value) * 255.0f), 255), 0, 0);
                cell.setFillColor(color);
            }
            cell.setPosition(x * cellSize, y * cellSize);
            window.draw(cell);
        }
    }

    window.display();
}

void setBoundary(std::vector<float>& field) {
    for (int x = 0; x < gridSizeX; ++x) {
        field[x] = field[gridSizeX + x];  // Top boundary
        field[(gridSizeY - 1) * gridSizeX + x] = field[(gridSizeY - 2) * gridSizeX + x];  // Bottom boundary
    }
    for (int y = 0; y < gridSizeY; ++y) {
        field[y * gridSizeX] = field[y * gridSizeX + 1];  // Left boundary
        field[y * gridSizeX + (gridSizeX - 1)] = field[y * gridSizeX + (gridSizeX - 2)];  // Right boundary
    }

    field[0] = 0.5f * (field[1] + field[gridSizeX]);

    field[gridSizeX - 1] = addNum((0.5f * field[gridSizeX - 2]), field[2 * gridSizeX - 1]);
    field[(gridSizeY - 1) * gridSizeX] = addNum((0.5f * field[(gridSizeY - 2) * gridSizeX]), (field[(gridSizeY - 1) * gridSizeX + 1]));
    field[(gridSizeY - 1) * gridSizeX + (gridSizeX - 1)] = addNum((0.5f * (field[(gridSizeY - 1) * gridSizeX + (gridSizeX - 2)])), (field[(gridSizeY - 2) * gridSizeX + (gridSizeX - 1)]));
}

void addBarrier(int x, int y, int width, int height) {
    for (int j = y; j < y + height; ++j) {
        for (int i = x; i < x + width; ++i)
            if (i >= 0 && i < gridSizeX && j >= 0 && j < gridSizeY) {
                barriers[j * gridSizeX + i] = true;
            }
    }
}