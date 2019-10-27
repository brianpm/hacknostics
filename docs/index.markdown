---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: simple site
tagline:  Atmospheric diagnostics and examples for your pivot to python
description: Provides a lot of examples in both plain scripts and notebooks that show how to do many common NCL tasks with python. We do this with relatively minimal packages, but we extensively use numpy, xarray, matplotlib, cartopy, and a little scipy.stats.
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>