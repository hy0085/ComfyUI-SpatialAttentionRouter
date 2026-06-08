/**
 * ColorRegion（颜色分区） — 内置画板前端 UI
 *
 * Features:
 * - Built-in HTML5 Canvas for drawing color masks
 * - Brush color auto-syncs with the selected region row
 * - Smart color management & Base64 sanitization
 * - Hijacks ALL native text widgets to prevent LiteGraph auto-stretch bugs
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

app.registerExtension({
    name: "ColorRegion.CanvasUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SpatialAttentionRouter") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            const colorPromptWidget = this.widgets.find(
                (w) => w.name === "color_prompts"
            );
            const canvasDataWidget = this.widgets.find(
                (w) => w.name === "canvas_data"
            );
            const globalPromptWidget = this.widgets.find(
                (w) => w.name === "global_prompt"
            );

            if (!colorPromptWidget) return r;

            // ── 1. Sanitize ──
            let initialText = colorPromptWidget.value || "";
            if (initialText.startsWith("data:image")) {
                initialText = "";
            }

            const seenColors = new Set();
            const cleanLines = [];
            initialText.split("\n").forEach((line) => {
                if (!line.trim()) return;
                const hex = (line.split(":")[0] || "")
                    .trim()
                    .toLowerCase();
                if (hex && !seenColors.has(hex)) {
                    seenColors.add(hex);
                    cleanLines.push(line);
                }
            });

            colorPromptWidget.value = cleanLines.join("\n");

            // ── 2. Hide ALL three native text widgets ──
            const hideWidget = (w) => {
                if (!w) return;
                w.type = "hidden";
                w.hidden = true;
                w.computeSize = () => [0, 0];
                if (w.element) {
                    w.element.style.display = "none";
                    w.element.style.opacity = "0";
                    w.element.style.height = "0";
                    w.element.style.pointerEvents = "none";
                }
                if (w.inputEl) {
                    w.inputEl.style.display = "none";
                    w.inputEl.style.height = "0";
                }
                if (
                    w.element &&
                    w.element.parentNode &&
                    w.element.parentNode.classList.contains(
                        "comfy-multiline-input"
                    )
                ) {
                    w.element.parentNode.style.display = "none";
                }
            };

            hideWidget(colorPromptWidget);
            hideWidget(canvasDataWidget);
            hideWidget(globalPromptWidget);

            // ── 3. Main container ──
            const container = $el("div", {
                style: {
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                    width: "100%",
                    marginTop: "4px",
                    paddingBottom: "8px",
                },
            });

            // ── 4. Custom global prompt (replaces native multiline) ──
            const globalPromptContainer = $el(
                "div",
                {
                    style: {
                        display: "flex",
                        flexDirection: "column",
                        gap: "4px",
                        marginBottom: "4px",
                    },
                },
                [
                    $el("label", {
                        textContent: "全局提示词（仅画质/风格）(Global Prompt, style only):",
                        style: {
                            color: "#ccc",
                            fontSize: "12px",
                            fontWeight: "bold",
                        },
                    }),
                    $el("textarea", {
                        placeholder:
                            "masterpiece, best quality...",
                        value: globalPromptWidget
                            ? globalPromptWidget.value
                            : "",
                        style: {
                            width: "100%",
                            height: "60px",
                            background: "#222",
                            color: "#fff",
                            border: "1px solid #444",
                            borderRadius: "4px",
                            padding: "6px",
                            resize: "vertical",
                            fontSize: "12px",
                            boxSizing: "border-box",
                            fontFamily: "inherit",
                        },
                        oninput: (e) => {
                            if (globalPromptWidget)
                                globalPromptWidget.value =
                                    e.target.value;
                        },
                    }),
                ]
            );
            container.appendChild(globalPromptContainer);

            // ── 5. Canvas ──
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.width = "100%";
            canvas.style.aspectRatio = "1/1";
            canvas.style.background = "#000";
            canvas.style.border = "2px solid #444";
            canvas.style.borderRadius = "8px";
            canvas.style.cursor = "crosshair";
            canvas.style.touchAction = "none";

            const ctx = canvas.getContext("2d");
            ctx.lineCap = "round";
            ctx.lineJoin = "round";

            if (
                canvasDataWidget &&
                canvasDataWidget.value &&
                canvasDataWidget.value.startsWith("data:image")
            ) {
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0);
                img.src = canvasDataWidget.value;
            } else {
                ctx.fillStyle = "#000000";
                ctx.fillRect(0, 0, 512, 512);
                if (canvasDataWidget)
                    canvasDataWidget.value =
                        canvas.toDataURL("image/png");
            }

            // ── 6. Drawing state ──
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            let activeColorIndex = 0;
            let brushSize = 30;

            const getLines = () =>
                (colorPromptWidget.value || "")
                    .split("\n")
                    .filter((l) => l.trim() !== "");

            const getActiveColor = () => {
                const lines = getLines();
                return lines.length === 0
                    ? "#ff0000"
                    : lines[activeColorIndex]?.split(":")[0] ||
                          "#ff0000";
            };

            const getCoords = (e) => {
                const rect = canvas.getBoundingClientRect();
                return [
                    (e.clientX - rect.left) *
                        (canvas.width / rect.width),
                    (e.clientY - rect.top) *
                        (canvas.height / rect.height),
                ];
            };

            const saveCanvas = () => {
                if (canvasDataWidget)
                    canvasDataWidget.value =
                        canvas.toDataURL("image/png");
            };

            canvas.addEventListener("pointerdown", (e) => {
                isDrawing = true;
                [lastX, lastY] = getCoords(e);
                canvas.setPointerCapture(e.pointerId);
                ctx.fillStyle = getActiveColor();
                ctx.beginPath();
                ctx.arc(
                    lastX,
                    lastY,
                    brushSize / 2,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            });

            canvas.addEventListener("pointermove", (e) => {
                if (!isDrawing) return;
                const [x, y] = getCoords(e);
                ctx.lineWidth = brushSize;
                ctx.strokeStyle = getActiveColor();
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.stroke();
                [lastX, lastY] = [x, y];
            });

            canvas.addEventListener("pointerup", (e) => {
                isDrawing = false;
                canvas.releasePointerCapture(e.pointerId);
                saveCanvas();
            });

            // ── 7. Toolbar ──
            const controlsRow = $el(
                "div",
                {
                    style: {
                        display: "flex",
                        gap: "10px",
                        alignItems: "center",
                        padding: "0 4px",
                    },
                },
                [
                    $el("label", {
                        textContent: "画笔大小 (Brush Size):",
                        style: { color: "#ccc", fontSize: "12px" },
                    }),
                    $el("input", {
                        type: "range",
                        min: "5",
                        max: "100",
                        value: brushSize,
                        style: { flex: 1 },
                        oninput: (e) => {
                            brushSize = parseInt(e.target.value);
                        },
                    }),
                    $el("button", {
                        textContent: "清除 (Clear)",
                        style: {
                            background: "#442222",
                            color: "#ff8888",
                            border: "none",
                            borderRadius: "4px",
                            padding: "4px 8px",
                            cursor: "pointer",
                            fontSize: "12px",
                        },
                        onclick: () => {
                            ctx.fillStyle = "#000000";
                            ctx.fillRect(0, 0, 512, 512);
                            saveCanvas();
                        },
                    }),
                ]
            );

            container.appendChild(canvas);
            container.appendChild(controlsRow);

            // ── 8. Region prompt list ──
            const promptsContainer = $el("div", {
                style: {
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                    marginTop: "8px",
                },
            });
            container.appendChild(promptsContainer);

            const updateLine = (index, newHex, newPrompt) => {
                const lines = getLines();
                lines[index] = `${newHex}: ${newPrompt}`;
                colorPromptWidget.value = lines.join("\n");
            };

            const resizeNode = () => {
                // Let LitheGraph recalculate from all widget computeSize
                // values (native + our custom), then apply.
                const out = this.computeSize();
                this.setSize([
                    Math.max(out[0], 420),
                    Math.max(out[1], 700),
                ]);
            };

            const renderUI = () => {
                promptsContainer.innerHTML = "";
                const lines = getLines();

                lines.forEach((line, index) => {
                    const parts = line.split(":");
                    const hex =
                        (parts[0] || "").trim() || "#ff0000";
                    const prompt = parts.slice(1).join(":").trim();
                    const isActive = index === activeColorIndex;

                    const row = $el(
                        "div",
                        {
                            style: {
                                display: "flex",
                                gap: "6px",
                                alignItems: "center",
                                padding: "6px",
                                background: isActive
                                    ? "rgba(100, 150, 200, 0.2)"
                                    : "transparent",
                                borderRadius: "6px",
                                border: isActive
                                    ? "1px solid #6699cc"
                                    : "1px solid transparent",
                                transition: "all 0.2s",
                            },
                        },
                        [
                            $el("input", {
                                type: "radio",
                                name: "active_brush_color",
                                checked: isActive,
                                style: {
                                    cursor: "pointer",
                                    width: "16px",
                                    height: "16px",
                                },
                                onchange: () => {
                                    activeColorIndex = index;
                                    renderUI();
                                },
                            }),
                            $el("input", {
                                type: "color",
                                value: hex,
                                style: {
                                    width: "26px",
                                    height: "26px",
                                    padding: "0",
                                    border: "none",
                                    cursor: "pointer",
                                    background: "transparent",
                                },
                                onchange: (e) => {
                                    updateLine(
                                        index,
                                        e.target.value,
                                        promptInput.value
                                    );
                                    if (isActive) renderUI();
                                },
                            }),
                            $el("input", {
                                type: "text",
                                value: prompt,
                                placeholder: "描述该区域 (describe region)...",
                                style: {
                                    flex: 1,
                                    height: "24px",
                                    background: "#222",
                                    color: "#fff",
                                    border: "1px solid #444",
                                    borderRadius: "4px",
                                    padding: "0 6px",
                                    fontSize: "12px",
                                },
                                oninput: (e) =>
                                    updateLine(
                                        index,
                                        hex,
                                        e.target.value
                                    ),
                            }),
                            $el("button", {
                                textContent: "✖",
                                style: {
                                    background: "#442222",
                                    color: "#ff8888",
                                    border: "none",
                                    borderRadius: "4px",
                                    padding: "4px 8px",
                                    cursor: "pointer",
                                },
                                onclick: () => {
                                    const l = getLines();
                                    l.splice(index, 1);
                                    colorPromptWidget.value =
                                        l.join("\n");
                                    if (
                                        activeColorIndex >=
                                        l.length
                                    )
                                        activeColorIndex = Math.max(
                                            0,
                                            l.length - 1
                                        );
                                    renderUI();
                                },
                            }),
                        ]
                    );
                    const promptInput = row.children[2];
                    promptsContainer.appendChild(row);
                });

                const presetColors = [
                    "#ff0000", "#00ff00", "#0000ff",
                    "#ffff00", "#ff00ff", "#00ffff",
                    "#ff8800", "#8800ff",
                ];
                const addBtn = $el("button", {
                    textContent: "添加新区域 (Add New Region)",
                    style: {
                        marginTop: "4px",
                        background: "#2a2a2a",
                        color: "#aaa",
                        border: "1px dashed #555",
                        borderRadius: "6px",
                        padding: "6px",
                        cursor: "pointer",
                        fontSize: "13px",
                        width: "100%",
                    },
                    onclick: () => {
                        const currentLines = getLines();
                        const usedColors = currentLines.map((l) =>
                            (l.split(":")[0] || "").trim().toLowerCase()
                        );

                        let nextColor = presetColors.find(
                            (c) => !usedColors.includes(c)
                        );
                        if (!nextColor)
                            nextColor =
                                "#" +
                                Math.floor(Math.random() * 16777215)
                                    .toString(16)
                                    .padStart(6, "0");

                        colorPromptWidget.value +=
                            (colorPromptWidget.value ? "\n" : "") +
                            `${nextColor}: `;
                        activeColorIndex = currentLines.length;
                        renderUI();
                    },
                });
                promptsContainer.appendChild(addBtn);

                resizeNode();
            };

            // ── 9. Register DOM widget ──
            const customWidget = this.addDOMWidget(
                "canvas_ui",
                "custom_ui",
                container,
                {
                    getValue: () => "",
                    setValue: () => {},
                }
            );

            // Report accurate HTML content height to LitheGraph
            customWidget.computeSize = function () {
                container.offsetHeight; // force reflow
                const h =
                    (globalPromptContainer?.offsetHeight || 0) +
                    (canvas?.offsetHeight || 0) +
                    (controlsRow?.offsetHeight || 0) +
                    (promptsContainer?.offsetHeight || 0) +
                    55;
                return [420, Math.max(h, 550)];
            };

            // DOM change watcher — triggers resize on add/remove region
            new MutationObserver(() => resizeNode()).observe(container, {
                childList: true,
                subtree: true,
            });

            setTimeout(() => {
                renderUI();
            }, 100);

            return r;
        };

        // ── Persistence: ensure global prompt survives save/load ──
        const origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            const r = origSerialize ? origSerialize.call(this, o) : o;
            const gpWidget = this.widgets?.find(
                (w) => w.name === "global_prompt"
            );
            console.log("[DEBUG] onSerialize gpWidget.value:", gpWidget?.value);
            if (gpWidget) {
                r.global_prompt_value = gpWidget.value || "";
            }
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origConfigure
                ? origConfigure.call(this, info)
                : undefined;
            const gpWidget = this.widgets?.find(
                (w) => w.name === "global_prompt"
            );
            console.log("[DEBUG] onConfigure gpWidget.value:", gpWidget?.value, "info:", info?.global_prompt_value);
            if (gpWidget && info?.global_prompt_value != null) {
                gpWidget.value = info.global_prompt_value;
                if (gpWidget.inputEl) {
                    gpWidget.inputEl.value = info.global_prompt_value;
                }
            }
            return r;
        };
    },
});
