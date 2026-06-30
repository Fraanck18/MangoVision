# Plan de Mejora de Diseño para ThermoMango

## Objetivo
Mejorar el diseño de la interfaz para que la página deje de parecer estática y se perciba como un dashboard operativo, dinámico y claro.

## 1. Dirección visual
- Convertirlo en una herramienta de monitoreo activo, no en una landing.
- Reforzar la sensación de sistema vivo con capas suaves, microinteracciones y feedback de estados.
- Mantener la seriedad técnica del producto: colores oscuros, acentos técnicos y claridad tipográfica.

## 2. Sistema de tokens y espaciado
- Definir una escala de espaciado clara y universal:
  - `--space-1: 8px`
  - `--space-2: 16px`
  - `--space-3: 24px`
  - `--space-4: 32px`
- Definir radios consistentes:
  - `--radius-sm: 8px`
  - `--radius-md: 12px`
  - `--radius-lg: 18px`
- Usar tokens tipográficos:
  - `--font-size-base: 15px`
  - `--text`: #1f2a27
  - `--text-muted`: #556e67
  - `--accent`: #1f9d6b

## 3. Jerarquía de contenido
- Titulares principales grandes y equilibrados con `clamp()`:
  - `font-size: clamp(1.6rem, 2.4vw, 2.4rem)`.
- Subtítulos compactos para separar secciones.
- Cuerpo legible con 65–75ch de longitud cuando haya texto largo.

## 4. Mejora de componentes clave
### 4.1 Barra lateral
- Estado activo más claro con fondo semitransparente y borde.
- Nav links con `hover` suave y `focus-visible` accesible.
- Usar iconos pequeños y texto que no se sienta bloqueado.

### 4.2 Tarjetas de indicadores (`.tarjeta-stat`)
- Quitar bordes rígidos y usar profundidad ligera.
- Aumentar padding a `var(--space-2)`.
- Mejorar el tamaño de valores y estados.
- Añadir guía visual de progreso / pequeños indicadores internos.

### 4.3 Botones
- `boton-primario`: fondo de acento, texto blanco, sombreado leve.
- `boton` secundario: borde suave, fondo `surface` con hover.
- `focus-visible` claro y consistente en ambos.

### 4.4 Paneles
- `panel` uniforme con padding `var(--space-2)` y borde tenue.
- Encabezado de panel con separación clara entre título y acciones.
- Usar microacciones con botones ghost para exportar / actualizar.

### 4.5 Controles de formulario
- Inputs con borde suave, padding consistente y `focus-visible` definido.
- Selects y switches con interacción evidente.

## 5. Movimiento y feedback
- Añadir transiciones suaves:
  - `transition: background-color .2s ease, transform .2s ease, box-shadow .2s ease`.
- Animar entrada de tarjetas con `opacity` y `translateY`.
- Animar el estado `en vivo` con `pulse` ligero en el indicador.
- Respetar `prefers-reduced-motion`.

## 6. Tema y apariencia
- Hacer que el tema `neon` sea una variante secundaria, no el default.
- Tema por defecto: oscuro técnico con acentos verdes y azules.
- El `neon` se usa solo en la HUD 3D o en situaciones de alerta.
- Asegurar que el tema claro/oscuro tenga contraste WCAG AA.

## 7. Estructura propuesta de cambios
- `web/css/styles.css`:
  - Consolidar tokens en `:root`
  - Refinar `.boton`, `.boton-primario`, `.nav-link`, `.tarjeta-stat`, `.panel`, `.selector-modos-dropdown`
  - Añadir reglas de `focus-visible`
  - Aplicar transiciones globales y `prefers-reduced-motion`
- `web/index.html`:
  - Mantener estructura actual pero usar `theme` claro como default al menos en la primera carga.
- `MangoVision/DESIGN.md`:
  - Documentar el sistema de tokens, colores y componentes.

## 8. Impacto esperado
- Menor sensación de “página fija”.
- Interfaz con jerarquía funcional más clara.
- Feedback visual en hover/focus que den sensación de control.
- Apariencia más moderna sin sacrificar la seriedad técnica.

## 9. Recomendación de implementación inmediata
1. Racionalizar los tokens de CSS.
2. Ajustar estado activo y hover en la navegación lateral.
3. Renovar las tarjetas de indicadores y los botones.
4. Limitar el tema `neon` a la HUD/3D.
5. Forzar `claro` temporalmente para verificar el diseño base.

---

### Comandos para subir el plan
```powershell
cd c:\Mango_Vision\MangoVision
git checkout -b feature/ui-improvement-plan
git add UI-IMPROVEMENT-PLAN.md
git commit -m "docs: add detailed UI improvement plan for ThermoMango"
git push -u origin feature/ui-improvement-plan
```
