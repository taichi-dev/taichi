---
sidebar_position: 6
---

# Documentation writing guide

Thank you for your contribution! This article briefly introduces syntax that will help you write documentation on this website. Note, the documentation is written in an extended version of [Markdown](https://daringfireball.net/projects/markdown/syntax), so for most of the time, you don't need special syntax besides the basic markdown syntax.

## 1. Insert code blocks

This site supports inserting code blocks with highlighted lines, for examples, the following:

````md
```python {1-2,4,6} title=snippet.py
@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02
```
````

will result in a code block like:

```python {1-2,4,6} title=snippet.py
@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02
```

## 2. Insert tables

```md
| Some Table Col 1 | Some Table Col 2 |
| :--------------: | :--------------: |
|       Val1       |       Val4       |
|       Val2       |       Val5       |
|       Val3       |       Val6       |
```

| Some Table Col 1 | Some Table Col 2 |
| :--------------: | :--------------: |
|       Val1       |       Val4       |
|       Val2       |       Val5       |
|       Val3       |       Val6       |

:::tip TIP
It's worth mentioning that [Tables Generator](https://www.tablesgenerator.com/markdown_tables) is a great tool for generating and re-formatting markdown tables.
:::

## 3. Cross-reference and anchor

To link to another section within the same article, you would use `[Return to ## 1. Insert code blocks](#1-insert-code-blocks)`: [Return to ## 1. Insert code blocks](#1-insert-code-blocks).

We follow the best practices suggested by [Docusaurus](https://docusaurus.io/docs/docs-markdown-features#referencing-other-documents) to cross-reference other documents, so to link to sections in other articles, please use the following relative-path based syntax, which
is docs-versioning and IDE/Github friendly:

- `[Return to Contribution guidelines](./contributor_guide.md)`: [Return to Contribution guidelines](./contributor_guide.md)
- `[Return to The Documentation which is at root](/docs/#portability)`: [Return to The Documentation](/docs/#portability)

## 4. Centered text block

To make a text or image block centered, use:

```md
<center>

Centered Text Block!

</center>
```

<center>

Centered Text Block!

</center>

:::danger NOTE
You **HAVE TO** insert blank lines to make them work:

```md
<center>

![](./some_pic.png)

</center>
```

:::

## 5. Text with color backgorund

You could use the following to highlight your text:

```html
<span id="inline-blue"> Text with blue background </span>,
<span id="inline-purple"> Text with purple background </span>,
<span id="inline-yellow"> Text with yellow background </span>,
<span id="inline-green"> Text with green background </span>
```

<span id="inline-blue"> Text with blue background </span>,
<span id="inline-purple"> Text with purple background </span>,
<span id="inline-yellow"> Text with yellow background </span>,
<span id="inline-green"> Text with green background </span>

## 6. Custom containers

As we already saw in this guide several places, we could add custom containers:

```md
:::tip
This is a tip without title!
:::
```

:::tip
This is a tip without title!
:::

```md
:::tip
This is a tip with a title!
:::
```

:::tip TITLE
This is a tip with a title!
:::

```md
:::note
This is a note!
:::
```

:::note
This is a note!
:::

```md caution
:::caution
This is a warning!
:::
```

:::caution WARNING
This is a warning!
:::

```md
:::danger DANGER
This is a danger!
:::
```

:::danger DANGER
This is a danger!
:::

## 7. Code groups

You could also insert tab-based code groups:

```markdown
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="apple"
  values={[
    {label: 'Apple', value: 'apple'},
    {label: 'Orange', value: 'orange'},
    {label: 'Banana', value: 'banana'},
  ]}>
  <TabItem value="apple">This is an apple 🍎</TabItem>
  <TabItem value="orange">This is an orange 🍊</TabItem>
  <TabItem value="banana">This is a banana 🍌</TabItem>
</Tabs>
```

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="apple"
  values={[
    {label: 'Apple', value: 'apple'},
    {label: 'Orange', value: 'orange'},
    {label: 'Banana', value: 'banana'},
  ]}>
  <TabItem value="apple">This is an apple 🍎</TabItem>
  <TabItem value="orange">This is an orange 🍊</TabItem>
  <TabItem value="banana">This is a banana 🍌</TabItem>
</Tabs>

## 8. Footnotes

It is important to cite the references, to do so, use the `markdown-it`'s footnotes syntax:

```md
This sentence has a footnote[^1]. (See footnote at the bottom of this guide.)

[^1]: I'm a footnote!
````

which results in:

---

This sentence has a footnote[^1]. (See footnote at the bottom of this guide.)

[^1]: I'm a footnote!

---

We could also write in-line footnotes, which is much easier to write without counting back and forth:

```md
This sentence has another footnote ^[I'm another footnote] (See footnote at the bottom of this page.)
```

which has the same effect:

---

This sentence has another footnote ^[I'm another footnote] (See footnote at the bottom of this page.)

---

## 9. Insert images

Insert images is as straight-forward as using the ordinary markdown syntax:

```md
![kernel](./life_of_kernel_lowres.jpg)
```

![kernel](./life_of_kernel_lowres.jpg)

## 10. Insert Table of Contents (ToC)

You could use:

```md
import TOCInline from '@theme/TOCInline';

<TOCInline toc={toc} />
```

to insert in-line ToC:

import TOCInline from '@theme/TOCInline';

<TOCInline toc={toc} />
