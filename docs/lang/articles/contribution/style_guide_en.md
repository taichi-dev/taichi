---
sidebar_position: 12
---

# Taichi Technical Documentation Style Guide

This is a reference for the developers and users at Taichi community to improve their writing and the consistency of Taichi's documentation. You can find detailed style, usage, and grammar in the following sections.

## General principles

### Style and tone

- Use active voice when possible.
- Write for scannability; use bullet points, short paragraphs, and sections/headers to break up your content.
- Oxford comma: In a list of three or more items, add a comma before the conjunction (for example: "Android, iOS, and Windows").
- Spell check your content.
- Be friendly by using "you".
- Review your content. Edit out any information your reader does not need to know.
- Remove ambiguity by choosing words with clear meaning.

Avoid the following:

- Exclamation marks, except in code snippets.
- Using phrases such as "simply" or "it is that simple" or "it is easy" in a procedure.
- Do not use dangling modifiers. A modifier "dangles" when the sentence is not clear about what is being modified.

### Write for a global audience

- Use simple verbs. For example, do not use words like utilize when the simpler word use conveys the same information.
- Do not use idiomatic or colloquial expressions.
- Avoid making negative voice constructions.
- Do not use double negative.
- Keep paragraphs short. Dense pages full of text are intimidating for readers.
- Address the reader directly by using “you” instead of “the developer”.
- Be inclusive in your writing. Use gender-neutral pronouns.
- Be consistent in your word usage. Do not use the same word to mean different things, and vice versa.

## Language and grammar

### Abbreviations and acronyms

Abbreviations are the shortened version of a word or phrase used to represent the whole. Examples include "s" for "seconds,” "approx." for "approximately," and "e.g." for "exempli gratia" (meaning "for example").
Abbreviations and acronyms can affect the clarity of Taichi content for the reader. While many are understood by our readers and do not need to be spelled out, for new or novel terms always spell out the first mention of an abbreviated term in the text, followed immediately by the abbreviation in parentheses . Use the abbreviated form for all subsequent references of the abbreviation on the same page.

#### Latin abbreviations

Do not use Latin abbreviations in your technical writing.
Many abbreviations derived from Latin are used in written English. Examples include "e.g." for "exempli gratia" (meaning "for example"), "i.e." for "id est" (meaning "in other words"), and "etc." for "et cetera" (meaning "and the other things").
Plain language principles suggest avoiding these abbreviated terms.

#### Contractions

Do not use contractions except in FAQs.
A contraction is a word or phrase that is shortened by dropping one or more letters. Examples include "aren't" for "are not", "let's" for "let us", and "can’t” for “cannot”. While any native English reader understands them, they add unnecessary complexity for non-native readers. For example, contractions that end with the letter "s" can be mistaken for possessive nouns. In business communication, the use of contractions is frowned upon as they make the tone of the writing too informal.
The only exception to this rule is when you are writing content for an FAQ. The more conversational tone of an FAQ allows for the use of contractions in titles and headers .

### Articles (a, an, the)

"A" and "an" are indefinite articles and are used before a singular countable noun. They refer to any member of a group. "The" is a definite article. It is used before singular and plural nouns and refers to one or more particular members of a group.
Sound rule for using "a" and "an"
The way a word or acronym is spoken determines whether "a" or "an" precedes it. Use "a" before a word that starts with a consonant sound, and "an" for words that begin with a vowel sound. For example "a URL", and "an SDK".

### Capitalization

- Use an uppercase letter to begin the first word of the text immediately following a colon.
- Use sentence case for captions and other figure-related text.
- Use sentence case for items in all types of lists.
- Use sentence case for all the elements in a table: contents, headings, labels, and captions.
- Refer to web page titles with the same casing used on the page.

### Ornamental words

An expletive is an added word or phrase that does not contribute meaning to the sentence. The most common expletives are "there are" and "there is".
- Not recommended: There are 10 users in the workspace.
- Recommended: The workspace has 10 users.

### Direct address or imperative mood
Use the imperative mood for task steps or a command line, a shell command for example.
Use third person singular for a description of an API method.
The imperative mood keeps the content concise. The direct address is more personal.
- Not recommended: Developers can download the SDK from here.
- Better: You can download the SDK from here.
- Recommended: Download the SDK from here.

### Gender-neutral

- Avoid using "his" or "her", or "he/she".
- Use the second person, "you", or the collective noun.

### Present tense

- Use the present tense as it creates concise sentences and provides a tone of immediacy. An exception to this rule is the release date of an SDK or other product. Always frame the release date in the past tense, as that tense will only be correct on the day of release. For example, use "v0.8.0 was released on September 23, 2021", NOT "v0.8.0 is released on September 23, 2021".
- Avoid using "will" unless you want to stress that something happens at a later point.
- Use future tense if there is a significant time delay that matters in the context.

### Second person

- In general, use second person "you" (sometimes implicit) in your docs.
- In glossary terms, avoid using person where possible. Use "developer" to refer to the reader if necessary.

### Clause order
- Put the most important information at the beginning of a sentence, followed by what the user can do with that information.
- Provide the context before you provide the instruction; that way, the reader can skip the additional information if it does not apply to their circumstance.

### Punctuations

#### Colons

- The first word after the colon should be in uppercase.
- When a colon introduces a list, the phrase before the colon must be a complete sentence.

#### Ampersands

- Do not use ampersands ("&") unless they are part of a name, UI string, or in a piece of code.

#### Hyphens

- All words following a hyphen are in lowercase, even if the word is at the beginning of a sentence. For example, "32-bit", or "Multi-threaded".
- Use a hyphen to form a compound adjective which is an adjective made up of more than one word. Examples include, "A 25-minute interval", "16-bit color", "a six-figure price", and more.
- Use a hyphen to indicate a common second element. For example, "a 10- to 11-digit number", "a six- or seven-hour delay", "a two-, three-, or fourfold increase".
- Many common prefixes, such as "co", "de", "pre", "pro", "re", and "sub" do not need a hyphen.
- Do not use a hyphen when forming a compound adjective with an adverb that ends in "ly".

#### Spaces

- Add a space before an opening parenthesis. Example: operating system (OS)
- Use only one space after full stops. Example: Add a space. One after the full stop.
- Use one space between numbers and units. Example: 256 Kbps.
- Use spaces around mathematical symbols. Example: V = off, width x height, x < y. Use spaces around dimensions. Example: 3.2 x 3.6 x 0.6 mm.
Note that million is not a unit, so there is no need to add a space between a number and M. For example, 10M is the right Taichi style.

## Plagiarism

Plagiarism puts the firm in a questionable position. Ensure that you do not copy and paste anything that you find from an online search to our technical documentation. As a tip, you can paraphrase contents that you find online.

## Formatting

### Headings and titles

Headings assist the reader in scanning content, helping them discover exactly what they are seeking. They provide structure and are visual points of reference for the reader.
Use headers to help outline your draft content. Some other points for consideration:
- Capitalize all words in a document title, except for articles and prepositions.
- Use sentence case for section titles.
- Be descriptive and concise.
- Focus on what the reader needs to know and what they can accomplish.
- Use ampersands or other symbols only if they appear in a UI or product name.
- Do NOT conclude a heading with a period or colon. (An exception are FAQs whose titles are often phrased as a conversation with the reader).

### Table headings

When referencing something specific (such as a unit of measure) in a table header, do not repeat it in the cells in that column. For example, if a table column header uses “Kbps”, then there is no need to repeat it in the cells for that column.

### Information

#### Note
Provides supplemental information that may not apply to all readers, but is important for those specific readers to know.
Wrap the notes in:
:::note
This is a note.
:::

#### Warning
Suggests proceeding with caution.
Wrap the notes in
:::caution WARNING
This is a warning.
:::

#### DANGER
Designed to guide the reader away from a circumstance that poses some form of problem or hazard.
Stronger than a Caution; it means "Don't do this."
Wrap the notes in:
:::danger DANGER
This is a danger!
:::

## Writing examples

### Example 1

- Not recommended: Taichi Zoo uses cookies for security, improvement and analytics purposes.
- Recommended: Taichi Zoo uses cookies for security, improvement, and analytics purposes.

**Comments:**
In a list of three or more items, add a comma before the conjunction。

### Example 2

- Not recommended: Two of the most commonly used types:
  - f32 represents a 32-bit floating point number.
  - i32 represents a 32-bit signed integer.
- Recommended: Two of the most commonly used types:
  - f32: 32-bit floating point number.
  - i32: 32-bit signed integer.

**Comments:**

Avoid repetitive information in a bullet list.

### Example 3

- Not recommended: If you run into this situation, Taichi's handy automatic differentiation (autodiff) system comes to the rescue!
- Recommended: Taichi's automatic differentiation (autodiff) system addresses this situation.

**Comments:**

- Avoid subjective descriptions, such as "handy" and "very", and dramatic expressions, for example "come to the rescue" in a technical document.

### Example 4

- Not recommended: ScopedProfileris used to analyze the performance of the Taichi JIT compiler (host).
- Recommended: ScopedProfiler analyzes the performance of the Taichi JIT compiler (host).

**Comments:**

- Use third person singular when describing a function, a method, or a callback.
- Use active voice as much as possible in a technical document.

### Example 5

- Not recommended: The easiest way is to make use of ti.GUI.
- Recommended: The easiest way is to use ti.GUI.

**Comments:**

Use simple verbs. A noun phrase, for example "make use of", is usually wordier than its original verb form, in this case "use".

### Example 6

- Not recommended: Use ti video -f40for creating a video with 40 FPS.
- Recommended:  Use ti video -f40to create a video with a frame rate of 40 FPS.

### Example 7

- Not recommended: Write less bugs.
- Recommended: Write fewer bugs.

**Comments:**

"Less" describes uncountable noun; "fewer" describes countable noun.

### Example 8

- Not recommended: Sometimes user may want to override the gradients provided by the Taichi autodiff system.
- Recommended: Sometimes you may want to override the gradients provided by the Taichi autodiff system.

**Comments:**

Address your audience directly by using "you".

### Example 9

- Not recommended: Compared to FLAT , query speed is much faster. Compared with IVFFLAT , less disk and CPU/GPU memory is required for IVF_SQ8.
- Recommended: IVF_SQ8 has a much higher query speed than FLAT, and requires less disk space and CPU/GPU memory than IVFFLAT.

**Comments:**

- IVF_SQ8 has a much faster query speed than FLAT (has). The second instance of "has" here can be omitted.
- "Compared to" and "Compared with" are usually wordy.

### Example 10

- Not recommended: Different from IVF_SQ8 , IVF_SQ8H uses a GPU-based coarse quantizer that greatly reduces the quantization time .
- Recommended: Unlike IVF_SQ8 , IVF_SQ8H uses a GPU-based coarse quantizer , which greatly reduces the time to quantize.

**Comments:**

- In technical writing, one word is always better than two.
- Which is used in a non-restrictive attributive clause; that is used in a restrictive attributive clause. Always precede a which-clause with a comma.


### Example 11

- Not recommended: When you use a client to update the following parameters, the updates take effect immediately.
- Recommended: Updates to the following parameters from a client take effect immediately:

**Comments:**

- The original is wordy.

### Example 12

- Not recommended: Vectors are quantized to 8-bit floats , which may cause accuracy loss.
- Recommended: Vectors are quantized to 8-bit floats. This may cause accuracy loss.
- Not recommended: However, the process to build a search graph requires a lot of computations for distances between vectors, which results in high computation costs.
- Recommended: However, the process of building a search graph requires a lot of computations for distances between vectors, resulting in high computation costs.

**Comments:**

- You cannot use which to refer to an entire preceding clause. Which only modifies the noun or noun phrase ( noun + prep. + noun) immediately preceding it. Use this to refer to the entire preceding clause .

### Example 13

- Not recommended: Make sure the Memory available to Docker Engine exceeds the sum of insert_buffer_size and cpu_cache_capacity you set in the config.yaml file.
- Recommended: Ensure that the Memory available to Docker Engine exceeds the sum of insert_buffer_size and cpu_cache_capacity , both of which are defined in config.yaml.

**Comments:**

- When it comes to technical writing, do not use more than one word when one word can convey the same information.
- Always use that to lead an appositive clause.
- If you have already spelt out the file name, you do not need to emphasize it is a file. Your readers can tell for themselves.

### Example 14

- Not recommended: Start the Prometheus server, with the --config.file flag pointing to the configuration file:
$ ./prometheus --config.file=prometheus.yml

- Recommended: Start the Prometheus server and specify the configuration file:
$ ./prometheus --config.file=prometheus.yml

**Comments:**

- Misuse of with. With modifies the subject of the main clause.
- The original is horribly specific. The revised version speaks for itself.

### Example 15

- Not recommended: This document talks about the following topics:
- Recommended: This document covers the following topics:

**Comments:**

- Anthropomorphism is not accepted in technical documents.

### Example 16

- Not recommended:
  - True: Enables the debug mode.
  - False: Disables the debug mode.
- Recommended:
  - True: Enable debug mode.
  - False: Disable debug mode.


**Comments:**

- Use imperative mood when desbribing a binary parameter; use third person singular when describing a function, a method, or a callback.
- Do not use the definite article before the word mode.

### Example 17

- Not recommended: This parameter is used to enable or disable Write Ahead Log (WAL).
- Recommended: This parameter enables or disables Write Ahead Log (WAL).

**Comments:**

- Clean, clear, and straight to the point!

### Example 18

- Not recommended: Active monitoring helps you identify problems early. But it is also essential to create alerting rules that promptly send notifications when there are events that require investigation or intervention.
- Recommended: Proactively monitoring metrics helps identify issues when they emerge. Creating alerting rules for events that require immediate intervention is essential as well.

**Comments:**

- Do not use "but" to lead a separate sentence.
- Way too many that-clauses!
- The "there be" construction is always awkward. An expletive is an added word or phrase that does not contribute meaning to the sentence. The most common expletives are "there are" and "there is".

### Example 19

- Not recommended: However, for delete operations, the operation speed is faster when write ahead log is enabled.
- Recommended: Delete operations are faster when write ahead log is enabled.

**Comments:**

- You cannot say  faster speed. You can say higher speed or greater speed . You can also say an operation is faster.
- The original is wordy.

## English style guide references

- Microsoft Writing Style Guide
- The Chicago Manual of Style
- Merriam-Webster's Dictionary
